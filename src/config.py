"""Configuration parameters for Smart Grid Energy Forecasting."""

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class GridForecastConfig:
    # Time-series resolution & horizons
    sampling_freq_hours: int = 1
    forecast_horizon_hours: int = 24
    history_days_train: int = 60

    # Quantiles for probabilistic uncertainty bounds
    quantiles: tuple = (0.10, 0.50, 0.90)  # P10 (base), P50 (median), P90 (peak reserve)

    # Feature engineering hyperparameters
    lags: tuple = (1, 2, 3, 24, 48, 168)
    rolling_windows: tuple = (6, 24)
    fourier_harmonics_daily: int = 3
    fourier_harmonics_weekly: int = 2

    # Storage paths
    artifacts_dir: Path = Path("artifacts")
    model_filename: str = "smart_grid_forecaster_bundle.joblib"


DEFAULT_GRID_CONFIG = GridForecastConfig()
