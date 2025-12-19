"""Pydantic schemas for the Smart Grid Probabilistic Forecasting API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class GridForecastRequest(BaseModel):
    horizon_hours: int = Field(default=24, ge=1, le=168, description="Forecast horizon (1 to 168 hours)")
    temperature_override_c: Optional[float] = Field(default=None, description="Optional what-if temperature override")
    custom_recent_net_load_mw: Optional[List[float]] = Field(default=None, description="Optional custom recent hourly meter readings")
    seed: int = Field(default=42, ge=0)


class HourlyQuantilePoint(BaseModel):
    step_ahead_hour: int
    timestamp_iso: Optional[str] = None
    p10_mw: float
    p50_mw: float
    p90_mw: float
    uncertainty_spread_mw: float


class GridForecastResponse(BaseModel):
    horizon_hours: int
    quantiles_modeled: List[str]
    forecast_points: List[HourlyQuantilePoint]
    peak_expected_p50_mw: float
    peak_reserve_p90_mw: float
    min_expected_p50_mw: float


class BacktestMetricsResponse(BaseModel):
    test_horizon_days: int
    mae_median_mw: float
    pinball_p10: float
    pinball_p50: float
    pinball_p90: float
    picp_coverage_pct: float
    mpiw_interval_width_mw: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    quantiles: List[float]
    horizon_hours_default: int
