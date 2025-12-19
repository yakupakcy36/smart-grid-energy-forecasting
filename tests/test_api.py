"""Integration tests for Smart Grid Forecasting FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert 0.5 in data["quantiles"]


def test_forecast_endpoint(client):
    payload = {
        "horizon_hours": 24,
        "temperature_override_c": 18.5,
        "custom_recent_net_load_mw": [410.0, 425.0, 430.0],
        "seed": 42
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["horizon_hours"] == 24
    assert len(data["forecast_points"]) == 24
    assert data["peak_reserve_p90_mw"] >= data["peak_expected_p50_mw"]
    assert data["forecast_points"][0]["timestamp_iso"] is not None

    # Verify monotonic quantile order for all points
    for pt in data["forecast_points"]:
        assert pt["p10_mw"] <= pt["p50_mw"] <= pt["p90_mw"]
        assert pt["uncertainty_spread_mw"] >= 0


def test_evaluate_backtest_endpoint(client):
    response = client.get("/evaluate/backtest")
    assert response.status_code == 200
    data = response.json()
    assert data["test_horizon_days"] == 7
    assert data["mae_median_mw"] > 0
    assert data["picp_coverage_pct"] > 50.0
