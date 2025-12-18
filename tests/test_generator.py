"""Unit tests for smart grid telemetry generator."""

import pandas as pd
from src.generator import generate_smart_grid_telemetry


def test_generator_columns_and_length():
    days = 10
    df = generate_smart_grid_telemetry(days=days, seed=42)
    assert len(df) == days * 24
    expected_cols = [
        "timestamp", "hour", "day_of_week", "temperature_c",
        "total_demand_mw", "solar_generation_mw", "wind_generation_mw", "net_load_mw"
    ]
    for col in expected_cols:
        assert col in df.columns


def test_generator_physical_bounds():
    df = generate_smart_grid_telemetry(days=5, seed=123)
    # Total demand must be strictly positive
    assert (df["total_demand_mw"] > 0).all()
    # Solar must be non-negative and zero during nighttime
    assert (df["solar_generation_mw"] >= 0).all()
    night_mask = (df["hour"] >= 21) | (df["hour"] <= 5)
    assert (df["solar_generation_mw"][night_mask] == 0.0).all()
    # Net load must equal total_demand - (solar + wind)
    expected_net = df["total_demand_mw"] - (df["solar_generation_mw"] + df["wind_generation_mw"])
    assert (abs(df["net_load_mw"] - expected_net) < 0.1).all()
