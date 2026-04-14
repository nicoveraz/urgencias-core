from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from urgencias_core.data.timeseries import ACUITY_LEVELS, hourly_timeseries


def test_hourly_shape_and_invariants(visits: pd.DataFrame) -> None:
    ts = hourly_timeseries(visits)

    expected_cols = {
        "timestamp",
        "arrivals",
        "departures",
        "occupancy",
        "mean_los_arriving",
    } | {f"occupancy_{a.lower()}" for a in ACUITY_LEVELS}
    assert expected_cols.issubset(set(ts.columns))

    assert (ts["occupancy"] >= 0).all()
    assert (ts["arrivals"] >= 0).all()
    assert (ts["departures"] >= 0).all()
    assert ts["arrivals"].sum() == len(visits)
    assert ts["departures"].sum() == len(visits)

    per_acuity = ts[[f"occupancy_{a.lower()}" for a in ACUITY_LEVELS]].sum(axis=1)
    np.testing.assert_allclose(per_acuity.to_numpy(), ts["occupancy"].to_numpy(), atol=1e-6)

    hours = pd.Series(ts["timestamp"]).diff().dropna().unique()
    assert len(hours) == 1 and hours[0] == np.timedelta64(1, "h")


def test_mean_los_arriving_matches_groupby(visits: pd.DataFrame) -> None:
    ts = hourly_timeseries(visits)
    bin_floor = visits["arrival"].dt.floor("h")
    expected = visits.groupby(bin_floor)["los_hours"].mean()
    got = ts.set_index("timestamp")["mean_los_arriving"].dropna()
    joined = expected.reindex(got.index)
    np.testing.assert_allclose(got.to_numpy(), joined.to_numpy(), rtol=1e-6)


def test_cache_roundtrip(visits: pd.DataFrame, tmp_path: Path) -> None:
    cache = tmp_path / "ts.parquet"
    first = hourly_timeseries(visits.head(500), cache_path=cache)
    assert cache.exists()
    second = hourly_timeseries(visits.head(500), cache_path=cache)
    pd.testing.assert_frame_equal(first, second)


def test_small_synthetic_case() -> None:
    visits = pd.DataFrame(
        {
            "arrival": pd.to_datetime(
                ["2024-01-01 10:00", "2024-01-01 10:30", "2024-01-01 12:00"]
            ),
            "discharge": pd.to_datetime(
                ["2024-01-01 11:30", "2024-01-01 13:00", "2024-01-01 13:00"]
            ),
            "acuity": ["C3", "C3", "C4"],
            "los_hours": [1.5, 2.5, 1.0],
        }
    )
    ts = hourly_timeseries(visits)
    ts = ts.set_index("timestamp")

    # At 10:00-11:00 census: patient1 present 10:00..11:00 (1h), patient2 present 10:30..11:00 (0.5h)
    # Mean census = (1*60 + 1*30) / 60 = 1.5
    assert ts.loc["2024-01-01 10:00", "occupancy"] == pytest.approx(1.5)
    assert ts.loc["2024-01-01 10:00", "arrivals"] == 2
    assert ts.loc["2024-01-01 12:00", "arrivals"] == 1
