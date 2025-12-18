"""Unit tests for multi-quantile model and loss metrics."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from src.config import DEFAULT_GRID_CONFIG
from src.generator import generate_smart_grid_telemetry
from src.model import (
    train_smart_grid_forecaster,
    compute_pinball_loss,
    SmartGridForecasterBundle,
)


def test_pinball_loss_exactness():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([90.0, 210.0])
    # For q = 0.5: loss is 0.5 * |diff|
    loss_p50 = compute_pinball_loss(y_true, y_pred, 0.5)
    expected_p50 = 0.5 * (10.0 + 10.0) / 2.0
    assert abs(loss_p50 - expected_p50) < 1e-4


def test_model_training_and_quantiles():
    df = generate_smart_grid_telemetry(days=25, seed=42)
    bundle, eval_res = train_smart_grid_forecaster(df, config=DEFAULT_GRID_CONFIG)

    assert eval_res.mae_median > 0
    assert 50.0 <= eval_res.picp_coverage_pct <= 99.0
    assert eval_res.mpiw_interval_width_mw > 0

    # Test serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "bundle.joblib"
        bundle.save(str(model_path))
        loaded = SmartGridForecasterBundle.load(str(model_path))
        assert len(loaded.models) == 3
