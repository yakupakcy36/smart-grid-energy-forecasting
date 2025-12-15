"""CLI runner for training and evaluating smart grid probabilistic forecasting."""

import argparse
from pathlib import Path
from src.config import DEFAULT_GRID_CONFIG
from src.generator import generate_smart_grid_telemetry
from src.model import train_smart_grid_forecaster, SmartGridForecasterBundle


def run_pipeline(days: int = 60, save_artifact: bool = True):
    print("=" * 65)
    print("SMART GRID PROBABILISTIC ENERGY FORECASTER: TRAINING & BENCHMARK")
    print("=" * 65)

    print(f"[1/4] Simulating {days} days of multi-bus smart grid telemetry...")
    df = generate_smart_grid_telemetry(days=days, seed=42)
    print(f"      Synthesized {len(df)} hourly timestamps")
    print(f"      Net Load mean: {df['net_load_mw'].mean():.2f} MW, min: {df['net_load_mw'].min():.2f} MW, max: {df['net_load_mw'].max():.2f} MW")

    print("\n[2/4] Training Multi-Quantile Regressors (P10, P50, P90) with HiGHS LP...")
    bundle, eval_res = train_smart_grid_forecaster(df, config=DEFAULT_GRID_CONFIG)

    print("\n[3/4] Backtest Performance on 7-Day Out-of-Sample Horizon:")
    print(f"      Median MAE (P50)             : {eval_res.mae_median:.2f} MW")
    print(f"      Pinball Loss P10             : {eval_res.pinball_p10:.2f}")
    print(f"      Pinball Loss P50             : {eval_res.pinball_p50:.2f}")
    print(f"      Pinball Loss P90             : {eval_res.pinball_p90:.2f}")
    print(f"      Empirical Coverage [P10-P90] : {eval_res.picp_coverage_pct:.1f}% (Nominal target: 80.0%)")
    print(f"      Mean Interval Width (MPIW)   : {eval_res.mpiw_interval_width_mw:.2f} MW")

    if save_artifact:
        out_path = DEFAULT_GRID_CONFIG.artifacts_dir / DEFAULT_GRID_CONFIG.model_filename
        print(f"\n[4/4] Serializing forecaster bundle to: {out_path}")
        bundle.save(str(out_path))

    print("\n[SUCCESS] Pipeline execution finished.")
    return bundle, eval_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Grid Forecasting CLI")
    parser.add_argument("--days", type=int, default=60, help="Days of telemetry data")
    args = parser.parse_args()
    run_pipeline(days=args.days)
