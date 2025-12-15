"""Multi-quantile probabilistic forecasting model for industrial smart grids."""

import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

from src.config import GridForecastConfig, DEFAULT_GRID_CONFIG
from src.features import GridFeatureExtractor


def compute_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Computes asymmetric quantile pinball loss."""
    diff = y_true - y_pred
    loss = np.maximum(quantile * diff, (quantile - 1.0) * diff)
    return float(np.mean(loss))


@dataclass
class ForecastEvaluationResult:
    mae_median: float
    pinball_p10: float
    pinball_p50: float
    pinball_p90: float
    picp_coverage_pct: float  # Percentage of true values within P10-P90
    mpiw_interval_width_mw: float  # Mean Prediction Interval Width


class SmartGridForecasterBundle:
    """Multi-quantile linear program forecaster with standardized feature pipeline."""

    def __init__(
        self,
        scaler: StandardScaler,
        models: Dict[float, QuantileRegressor],
        feature_names: List[str],
        config: GridForecastConfig = DEFAULT_GRID_CONFIG,
    ):
        self.scaler = scaler
        self.models = models
        self.feature_names = feature_names
        self.config = config

    def predict_quantiles(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generates P10, P50, and P90 quantile forecasts."""
        X_aligned = X[self.feature_names]
        X_scaled = self.scaler.transform(X_aligned)

        predictions: Dict[str, np.ndarray] = {}
        for q, model in self.models.items():
            key = f"p{int(q * 100)}"
            preds = model.predict(X_scaled)
            predictions[key] = np.round(preds, 2)

        # Enforce monotonic quantile consistency: P10 <= P50 <= P90
        if "p10" in predictions and "p50" in predictions:
            predictions["p50"] = np.maximum(predictions["p10"], predictions["p50"])
        if "p50" in predictions and "p90" in predictions:
            predictions["p90"] = np.maximum(predictions["p50"], predictions["p90"])

        return predictions

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> ForecastEvaluationResult:
        """Evaluates probabilistic forecasts against held-out grid telemetry."""
        preds = self.predict_quantiles(X_test)
        y_true = y_test.to_numpy()

        p10 = preds["p10"]
        p50 = preds["p50"]
        p90 = preds["p90"]

        mae = float(np.mean(np.abs(y_true - p50)))
        loss_p10 = compute_pinball_loss(y_true, p10, 0.10)
        loss_p50 = compute_pinball_loss(y_true, p50, 0.50)
        loss_p90 = compute_pinball_loss(y_true, p90, 0.90)

        # Coverage probability: fraction inside [p10, p90]
        inside = (y_true >= p10) & (y_true <= p90)
        coverage_pct = float(np.mean(inside) * 100.0)

        # Mean Prediction Interval Width
        mpiw = float(np.mean(p90 - p10))

        return ForecastEvaluationResult(
            mae_median=round(mae, 2),
            pinball_p10=round(loss_p10, 2),
            pinball_p50=round(loss_p50, 2),
            pinball_p90=round(loss_p90, 2),
            picp_coverage_pct=round(coverage_pct, 1),
            mpiw_interval_width_mw=round(mpiw, 2),
        )

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "SmartGridForecasterBundle":
        return joblib.load(filepath)


def train_smart_grid_forecaster(
    telemetry_df: pd.DataFrame,
    config: GridForecastConfig = DEFAULT_GRID_CONFIG,
) -> Tuple[SmartGridForecasterBundle, ForecastEvaluationResult]:
    """Trains multi-quantile regressor across historical smart grid telemetry."""
    extractor = GridFeatureExtractor(config=config)
    X, y = extractor.create_features(telemetry_df)

    # Chronological Train / Test split (Last 7 days held out for backtesting)
    split_idx = len(X) - (7 * 24)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models: Dict[float, QuantileRegressor] = {}
    for q in config.quantiles:
        # Alpha L1 regularization for sparse, stable coefficient vectors
        reg = QuantileRegressor(quantile=q, alpha=0.05, solver="highs")
        reg.fit(X_train_scaled, y_train.to_numpy())
        models[q] = reg

    bundle = SmartGridForecasterBundle(
        scaler=scaler,
        models=models,
        feature_names=list(X.columns),
        config=config,
    )

    eval_result = bundle.evaluate(X_test, y_test)
    return bundle, eval_result
