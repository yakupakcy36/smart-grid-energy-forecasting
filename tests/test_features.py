"""Unit tests for feature engineering pipeline."""

import numpy as np
from src.config import DEFAULT_GRID_CONFIG
from src.generator import generate_smart_grid_telemetry
from src.features import GridFeatureExtractor


def test_feature_extraction_pipeline():
    df = generate_smart_grid_telemetry(days=20, seed=42)
    extractor = GridFeatureExtractor(config=DEFAULT_GRID_CONFIG)
    X, y = extractor.create_features(df)

    assert len(X) == len(y)
    assert len(X) > 0
    # Check that lag columns exist
    for lag in DEFAULT_GRID_CONFIG.lags:
        assert f"lag_{lag}h" in X.columns
    # Check that Fourier terms exist
    assert "fourier_daily_sin_1" in X.columns
    assert "fourier_weekly_sin_1" in X.columns
    # No NaNs or Infs allowed
    assert not X.isna().any().any()
    assert not y.isna().any()
