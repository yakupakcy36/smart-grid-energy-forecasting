"""FastAPI application for Smart Grid Energy Probabilistic Forecasting."""

from contextlib import asynccontextmanager
from typing import Optional, List
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from src.config import DEFAULT_GRID_CONFIG
from src.generator import generate_smart_grid_telemetry
from src.features import GridFeatureExtractor
from src.model import train_smart_grid_forecaster, SmartGridForecasterBundle, ForecastEvaluationResult
from api.schemas import (
    GridForecastRequest,
    GridForecastResponse,
    HourlyQuantilePoint,
    BacktestMetricsResponse,
    HealthResponse,
)

bundle: Optional[SmartGridForecasterBundle] = None
cached_eval: Optional[ForecastEvaluationResult] = None
baseline_df: Optional[pd.DataFrame] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle, cached_eval, baseline_df
    # Pre-generate telemetry for serving and training
    baseline_df = generate_smart_grid_telemetry(days=DEFAULT_GRID_CONFIG.history_days_train, seed=42)
    artifact_path = DEFAULT_GRID_CONFIG.artifacts_dir / DEFAULT_GRID_CONFIG.model_filename

    if artifact_path.exists():
        try:
            bundle = SmartGridForecasterBundle.load(str(artifact_path))
        except Exception:
            bundle, cached_eval = train_smart_grid_forecaster(baseline_df, config=DEFAULT_GRID_CONFIG)
    else:
        bundle, cached_eval = train_smart_grid_forecaster(baseline_df, config=DEFAULT_GRID_CONFIG)
    yield


app = FastAPI(
    title="Smart Grid Probabilistic Energy Forecasting API",
    description=(
        "Production multi-quantile energy load & renewable balance forecasting engine. "
        "Provides P10, P50 (median), and P90 (peak reserve) uncertainty intervals using "
        "linear programming quantile regression with Fourier seasonality and lag features."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health():
    return HealthResponse(
        status="healthy",
        model_loaded=(bundle is not None),
        quantiles=list(DEFAULT_GRID_CONFIG.quantiles),
        horizon_hours_default=DEFAULT_GRID_CONFIG.forecast_horizon_hours,
    )


@app.post("/forecast", response_model=GridForecastResponse, tags=["Forecasting"])
def forecast(req: GridForecastRequest):
    """Generates probabilistic multi-quantile load predictions across the requested horizon."""
    if bundle is None or baseline_df is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model bundle not ready")

    # Use latest historical telemetry
    df = baseline_df.copy()
    if req.temperature_override_c is not None:
        df["temperature_c"] = req.temperature_override_c

    extractor = GridFeatureExtractor(config=DEFAULT_GRID_CONFIG)
    X, _ = extractor.create_features(df)

    # Slice the latest requested horizon rows
    horizon = min(req.horizon_hours, len(X))
    X_slice = X.iloc[-horizon:].copy().reset_index(drop=True)

    preds = bundle.predict_quantiles(X_slice)
    p10_arr = preds["p10"]
    p50_arr = preds["p50"]
    p90_arr = preds["p90"]

    points: List[HourlyQuantilePoint] = []
    for step in range(len(p50_arr)):
        p10 = float(p10_arr[step])
        p50 = float(p50_arr[step])
        p90 = float(p90_arr[step])
        points.append(
            HourlyQuantilePoint(
                step_ahead_hour=step + 1,
                p10_mw=round(p10, 2),
                p50_mw=round(p50, 2),
                p90_mw=round(p90, 2),
                uncertainty_spread_mw=round(p90 - p10, 2),
            )
        )

    return GridForecastResponse(
        horizon_hours=horizon,
        quantiles_modeled=["P10", "P50", "P90"],
        forecast_points=points,
        peak_expected_p50_mw=round(float(np.max(p50_arr)), 2),
        peak_reserve_p90_mw=round(float(np.max(p90_arr)), 2),
        min_expected_p50_mw=round(float(np.min(p50_arr)), 2),
    )


@app.get("/evaluate/backtest", response_model=BacktestMetricsResponse, tags=["Diagnostics"])
def evaluate_backtest():
    """Returns out-of-sample backtest metrics on held-out test data."""
    if bundle is None or baseline_df is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model bundle not ready")

    global cached_eval
    if cached_eval is None:
        extractor = GridFeatureExtractor(config=DEFAULT_GRID_CONFIG)
        X, y = extractor.create_features(baseline_df)
        split_idx = len(X) - (7 * 24)
        cached_eval = bundle.evaluate(X.iloc[split_idx:], y.iloc[split_idx:])

    return BacktestMetricsResponse(
        test_horizon_days=7,
        mae_median_mw=cached_eval.mae_median,
        pinball_p10=cached_eval.pinball_p10,
        pinball_p50=cached_eval.pinball_p50,
        pinball_p90=cached_eval.pinball_p90,
        picp_coverage_pct=cached_eval.picp_coverage_pct,
        mpiw_interval_width_mw=cached_eval.mpiw_interval_width_mw,
    )
