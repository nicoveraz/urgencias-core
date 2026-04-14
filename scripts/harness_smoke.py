"""Phase-3 smoke test: run the full baseline + candidate battery on the
synthetic fixture and print harness results.

Usage:
    uv run python scripts/harness_smoke.py
    uv run python scripts/harness_smoke.py --target arrivals --holdout 168
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from urgencias_core.data.loader import load_visits
from urgencias_core.data.timeseries import hourly_timeseries
from urgencias_core.eval.baselines import (
    SeasonalNaiveBaseline,
    auto_arima,
    auto_ets,
    auto_theta,
    mstl,
)
from urgencias_core.eval.harness import run_harness
from urgencias_core.models.lgb_quantile import LGBQuantileForecaster
from urgencias_core.models.protocol import HorizonSpec

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic_ed_visits.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="arrivals", help="Column to forecast")
    parser.add_argument("--holdout", type=int, default=168, help="Holdout length in hours")
    parser.add_argument(
        "--train-window",
        type=int,
        default=24 * 90,
        help="Number of training hours to use (recent window). Default 90 days.",
    )
    args = parser.parse_args()

    print("Loading fixture...")
    visits = load_visits(FIXTURE)
    print(f"  {len(visits):,} visits")

    print("Computing hourly timeseries...")
    t0 = time.time()
    series = hourly_timeseries(visits)
    print(f"  {len(series):,} hours ({time.time() - t0:.1f}s)")

    if args.train_window and args.train_window > 0:
        series = series.iloc[-args.train_window :].reset_index(drop=True)
        print(f"  Using recent training window: {len(series):,} hours")

    baselines = {
        "SeasonalNaive": SeasonalNaiveBaseline(),
        "AutoARIMA": auto_arima(season_length=24),
        "AutoETS": auto_ets(season_length=24),
        "AutoTheta": auto_theta(season_length=24),
        "MSTL": mstl(season_length=(24, 168)),
    }
    candidates = {
        "LGBQuantile": LGBQuantileForecaster(n_estimators=400),
    }

    horizon = HorizonSpec(grain="h", length=args.holdout)
    print(
        f"\nRunning harness: target={args.target!r}, holdout={args.holdout}h, "
        f"train={len(series) - args.holdout}h"
    )
    t0 = time.time()
    report = run_harness(
        series=series,
        target_col=args.target,
        horizon=horizon,
        holdout_length=args.holdout,
        baselines=baselines,
        candidates=candidates,
    )
    print(f"\nHarness wall time: {time.time() - t0:.1f}s")
    print(f"Warning triggered: {report.warning_triggered}")


if __name__ == "__main__":
    main()
