"""Physics-informed synthetic smart grid telemetry generator."""

from datetime import datetime, timedelta
from typing import Tuple
import numpy as np
import pandas as pd


def generate_smart_grid_telemetry(
    start_date: str = "2025-01-01 00:00:00",
    days: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates realistic hourly multi-component grid time-series:
    - Base demand (industrial shift profiles + residential cycles)
    - Outdoor ambient temperature (cooling/heating load drivers)
    - Solar PV generation (diurnal parabolic arc with cloud intermittency)
    - Wind turbine power (Weibull distribution with cut-in / rated speeds)
    - Net grid load = Total Demand - (Solar + Wind)
    """
    np.random.seed(seed)
    n_hours = days * 24
    timestamps = [datetime.fromisoformat(start_date) + timedelta(hours=h) for h in range(n_hours)]

    hours = np.array([ts.hour for ts in timestamps])
    day_of_week = np.array([ts.weekday() for ts in timestamps])
    day_of_year = np.array([ts.timetuple().tm_yday for ts in timestamps])

    # 1. Ambient Temperature Profile (in Celsius)
    # Seasonal base + daily diurnal oscillation + autoregressive weather noise
    t_seasonal = 14.0 + 10.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365.25)
    t_diurnal = 5.0 * np.sin(2 * np.pi * (hours - 9) / 24.0)
    weather_noise = np.zeros(n_hours)
    noise_val = 0.0
    for h in range(n_hours):
        noise_val = 0.92 * noise_val + np.random.normal(0, 0.8)
        weather_noise[h] = noise_val
    temperature = t_seasonal + t_diurnal + weather_noise

    # 2. Base Power Demand (MW)
    # Baseline load ~ 450 MW
    base_load = 450.0

    # Diurnal demand curve (morning peak at 09:00, evening peak at 19:00, night trough at 04:00)
    diurnal_demand = 85.0 * np.sin(2 * np.pi * (hours - 5) / 24.0) + 35.0 * np.sin(4 * np.pi * (hours - 5) / 24.0)

    # Weekend reduction (-22% industrial activity)
    is_weekend = (day_of_week >= 5).astype(float)
    weekend_factor = 1.0 - 0.22 * is_weekend

    # Temperature-sensitive heating and cooling loads (degree days effect)
    cooling_demand = np.maximum(0, temperature - 22.0) * 12.0
    heating_demand = np.maximum(0, 12.0 - temperature) * 8.0

    load_noise = np.random.normal(0, 15.0, size=n_hours)
    total_demand = (base_load + diurnal_demand) * weekend_factor + cooling_demand + heating_demand + load_noise
    total_demand = np.clip(total_demand, 200.0, 950.0)

    # 3. Solar PV Generation (MW)
    # Solar capacity: 180 MW peak. Only generates between 06:00 and 20:00.
    solar_gen = np.zeros(n_hours)
    daylight_mask = (hours >= 6) & (hours <= 20)
    solar_hour_angle = np.pi * (hours[daylight_mask] - 6) / 14.0
    raw_solar = 180.0 * np.sin(solar_hour_angle) ** 1.5
    cloud_attenuation = np.clip(np.random.beta(5, 2, size=len(raw_solar)), 0.15, 1.0)
    solar_gen[daylight_mask] = np.maximum(0.0, raw_solar * cloud_attenuation)

    # 4. Wind Generation (MW)
    # Wind capacity: 220 MW peak. Weibull wind speeds transformed by cubic power curve
    wind_speeds = np.random.weibull(2.1, size=n_hours) * 7.5  # m/s
    wind_gen = np.zeros(n_hours)
    # Cut-in: 3 m/s, Rated: 12 m/s, Cut-out: 25 m/s
    operating_wind = (wind_speeds >= 3.0) & (wind_speeds <= 25.0)
    wind_gen[operating_wind] = 220.0 * np.clip((wind_speeds[operating_wind] - 3.0) / (12.0 - 3.0), 0.0, 1.0) ** 3
    wind_gen[wind_speeds > 12.0] = 220.0
    wind_gen[wind_speeds > 25.0] = 0.0

    # 5. Net Grid Load
    net_load = total_demand - (solar_gen + wind_gen)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "hour": hours,
            "day_of_week": day_of_week,
            "temperature_c": np.round(temperature, 2),
            "total_demand_mw": np.round(total_demand, 2),
            "solar_generation_mw": np.round(solar_gen, 2),
            "wind_generation_mw": np.round(wind_gen, 2),
            "net_load_mw": np.round(net_load, 2),
        }
    )
    return df
