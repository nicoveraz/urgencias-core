"""Generate the synthetic ED visit fixture used by tests, demos, and CI.

Writes ~40k rows spanning 2022-01-01 to 2024-12-31 with realistic Chilean
emergency department patterns:

- Day-of-week seasonality (higher on Mon, lower Sun)
- Hour-of-day arrivals curve (morning ramp, evening peak)
- Acuity distribution dominated by C3/C4, smaller C1/C2/C5 tails
- LOS conditional on acuity (C1/C2 long, C5 short)
- Strong winter respiratory peak (Jun-Aug in southern Chile)
- Holiday and Christmas/New Year bumps
- Occupancy emerges naturally from (arrivals, LOS), lagging arrivals by ~LOS

Run with:
    uv run python scripts/generate_fixture.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 20260413

ACUITY_PROBS = {
    "C1": 0.02,
    "C2": 0.10,
    "C3": 0.38,
    "C4": 0.42,
    "C5": 0.08,
}

# Log-normal LOS parameters per acuity (mu, sigma) in log-hours.
# Median hours ~ exp(mu). C1/C2 include boarding tail.
LOS_LOGNORMAL = {
    "C1": (1.80, 0.80),   # median ~6.0h, long tail (critical + boarding)
    "C2": (1.40, 0.75),   # median ~4.0h
    "C3": (0.90, 0.60),   # median ~2.5h
    "C4": (0.40, 0.55),   # median ~1.5h
    "C5": (-0.10, 0.50),  # median ~0.9h
}

DISPOSITIONS = {
    "C1": {"hospitalizacion": 0.55, "traslado": 0.15, "alta": 0.28, "fallecido": 0.02},
    "C2": {"hospitalizacion": 0.35, "traslado": 0.05, "alta": 0.59, "fallecido": 0.01},
    "C3": {"hospitalizacion": 0.10, "traslado": 0.01, "alta": 0.89, "fallecido": 0.0},
    "C4": {"hospitalizacion": 0.02, "traslado": 0.0,  "alta": 0.98, "fallecido": 0.0},
    "C5": {"hospitalizacion": 0.005, "traslado": 0.0, "alta": 0.995, "fallecido": 0.0},
}

DOW_MULTIPLIER = np.array([1.15, 1.05, 1.00, 1.00, 1.05, 0.95, 0.85])  # Mon..Sun

# Hour-of-day arrival shape, 24 values summing to 1. Morning ramp from ~6am,
# midday plateau, secondary evening peak, quiet overnight.
HOUR_WEIGHTS = np.array(
    [0.020, 0.015, 0.012, 0.010, 0.010, 0.015,
     0.025, 0.040, 0.055, 0.065, 0.070, 0.070,
     0.065, 0.060, 0.055, 0.055, 0.055, 0.060,
     0.065, 0.060, 0.050, 0.045, 0.035, 0.028]
)
HOUR_WEIGHTS = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()

BASE_DAILY_ARRIVALS = 36.0  # average per day before seasonality


def winter_multiplier(day_of_year: np.ndarray) -> np.ndarray:
    """Winter respiratory peak centered around day 196 (mid-July, southern Chile)."""
    radians = 2 * np.pi * (day_of_year - 196) / 365.25
    return 1.0 + 0.35 * np.cos(radians)  # range ~0.65 (summer) to 1.35 (winter)


def holiday_multiplier(dates: pd.DatetimeIndex) -> np.ndarray:
    """Bump for holidays/Christmas/New Year (presentations rise a bit, not fall).

    Rationale: in Chilean EDs, holidays bring family-gathering related
    presentations and reduced primary-care availability.
    """
    try:
        import holidays as pyholidays

        cl = pyholidays.country_holidays("CL", years=range(dates.year.min(), dates.year.max() + 1))
    except Exception:
        cl = {}
    mult = np.ones(len(dates))
    for i, d in enumerate(dates):
        if d.date() in cl:
            mult[i] = 1.15
        if (d.month == 12 and d.day in (24, 25, 31)) or (d.month == 1 and d.day == 1):
            mult[i] = 1.25
    return mult


def generate(start: str, end: str, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range(start=start, end=end, freq="D")

    dow = days.dayofweek.to_numpy()
    doy = days.dayofyear.to_numpy()

    lam = (
        BASE_DAILY_ARRIVALS
        * DOW_MULTIPLIER[dow]
        * winter_multiplier(doy)
        * holiday_multiplier(days)
    )
    daily_counts = rng.poisson(lam)

    total = int(daily_counts.sum())
    arrival_day_idx = np.repeat(np.arange(len(days)), daily_counts)
    hours = rng.choice(24, size=total, p=HOUR_WEIGHTS)
    minutes = rng.integers(0, 60, size=total)
    seconds = rng.integers(0, 60, size=total)

    arrival = (
        days[arrival_day_idx]
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
        + pd.to_timedelta(seconds, unit="s")
    )

    acuity_values = np.array(list(ACUITY_PROBS.keys()))
    acuity_probs = np.array(list(ACUITY_PROBS.values()))
    acuity = rng.choice(acuity_values, size=total, p=acuity_probs)

    los_hours = np.empty(total)
    for code, (mu, sigma) in LOS_LOGNORMAL.items():
        mask = acuity == code
        n = int(mask.sum())
        if n:
            los_hours[mask] = rng.lognormal(mean=mu, sigma=sigma, size=n)
    los_hours = np.clip(los_hours, 1 / 60, 120.0)  # 1 min to 5 days

    discharge = arrival + pd.to_timedelta(los_hours, unit="h")

    disposition = np.empty(total, dtype=object)
    for code, probs in DISPOSITIONS.items():
        mask = acuity == code
        n = int(mask.sum())
        if n:
            disposition[mask] = rng.choice(
                list(probs.keys()), size=n, p=list(probs.values())
            )

    df = pd.DataFrame(
        {
            "visit_id": np.arange(total, dtype=np.int64),
            "arrival": arrival,
            "discharge": discharge,
            "acuity": acuity,
            "disposition": disposition,
            "los_hours": los_hours,
        }
    ).sort_values("arrival").reset_index(drop=True)
    df["visit_id"] = np.arange(len(df), dtype=np.int64)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD). Mutually exclusive with --years.")
    parser.add_argument(
        "--years",
        type=int,
        default=None,
        help="Span length in whole years, ending on Dec 31 of start_year + years - 1. "
        "Mutually exclusive with --end.",
    )
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "synthetic_ed_visits.parquet",
    )
    args = parser.parse_args()

    if args.end and args.years:
        parser.error("--end and --years are mutually exclusive")
    if args.years is not None:
        start = pd.Timestamp(args.start)
        end_str = f"{start.year + args.years - 1}-12-31"
    else:
        end_str = args.end or "2024-12-31"

    df = generate(args.start, end_str, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"Wrote {len(df):,} rows to {args.out}")
    print(f"Date range: {df['arrival'].min()} -> {df['arrival'].max()}")
    print("Acuity distribution:")
    print(df["acuity"].value_counts(normalize=True).round(3).to_string())
    print(f"Median LOS (h): {df['los_hours'].median():.2f}")


if __name__ == "__main__":
    main()
