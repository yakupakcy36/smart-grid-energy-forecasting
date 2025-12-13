"""Feature engineering for probabilistic power grid time-series forecasting."""

from typing import List, Tuple
import numpy as np
import pandas as pd
from src.config import GridForecastConfig, DEFAULT_GRID_CONFIG


class GridFeatureExtractor:
    """Builds lag, rolling statistics, Fourier harmonics, and calendar features."""

    def __init__(self, config: GridForecastConfig = DEFAULT_GRID_CONFIG):
        self.config = config

    def create_features(self, df: pd.DataFrame, target_col: str = "net_load_mw") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transforms raw hourly telemetry into feature matrix and target series.
        Drops incomplete initial rows caused by lag windows.
        """
        data = df.copy()

        # 1. Calendar & Cyclical Hour / Day features
        data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0)
        data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7.0)
        data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7.0)
        data["is_weekend"] = (data["day_of_week"] >= 5).astype(float)

        # 2. Fourier Terms for Multi-Seasonal Oscillations
        # Daily harmonics (T = 24)
        for k in range(1, self.config.fourier_harmonics_daily + 1):
            data[f"fourier_daily_sin_{k}"] = np.sin(2 * np.pi * k * data["hour"] / 24.0)
            data[f"fourier_daily_cos_{k}"] = np.cos(2 * np.pi * k * data["hour"] / 24.0)

        # Weekly harmonics (T = 168)
        hour_of_week = data["day_of_week"] * 24 + data["hour"]
        for k in range(1, self.config.fourier_harmonics_weekly + 1):
            data[f"fourier_weekly_sin_{k}"] = np.sin(2 * np.pi * k * hour_of_week / 168.0)
            data[f"fourier_weekly_cos_{k}"] = np.cos(2 * np.pi * k * hour_of_week / 168.0)

        # 3. Autoregressive Lag Features
        for lag in self.config.lags:
            data[f"lag_{lag}h"] = data[target_col].shift(lag)

        # 4. Rolling Window Statistics (shifted by 1 to prevent data leakage)
        for window in self.config.rolling_windows:
            rolled = data[target_col].shift(1).rolling(window=window)
            data[f"rolling_mean_{window}h"] = rolled.mean()
            data[f"rolling_std_{window}h"] = rolled.std()
            data[f"rolling_min_{window}h"] = rolled.min()
            data[f"rolling_max_{window}h"] = rolled.max()

        # 5. Exogenous Meteorological Regressors
        if "temperature_c" in data.columns:
            data["cooling_degree"] = np.maximum(0.0, data["temperature_c"] - 22.0)
            data["heating_degree"] = np.maximum(0.0, 12.0 - data["temperature_c"])

        # Drop initial NaN rows created by maximum lag (168 hours = 1 week)
        feature_cols = [
            c
            for c in data.columns
            if c not in ["timestamp", "total_demand_mw", "solar_generation_mw", "wind_generation_mw", "net_load_mw"]
        ]

        clean_data = data.dropna().reset_index(drop=True)
        X = clean_data[feature_cols]
        y = clean_data[target_col]

        return X, y
