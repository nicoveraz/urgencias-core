from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from urgencias_core.data.loader import REQUIRED_COLUMNS, SchemaError, load_visits


def test_load_fixture(fixture_path: Path) -> None:
    df = load_visits(fixture_path)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    assert "los_hours" in df.columns
    assert len(df) > 10_000
    assert df["arrival"].is_monotonic_increasing
    assert (df["discharge"] > df["arrival"]).all()
    assert df["acuity"].isin({"C1", "C2", "C3", "C4", "C5"}).all()


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_visits(tmp_path / "nope.parquet")


def test_missing_column_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.parquet"
    pd.DataFrame({"visit_id": [1], "arrival": [pd.Timestamp("2024-01-01")]}).to_parquet(bad)
    with pytest.raises(SchemaError, match="Missing required columns"):
        load_visits(bad)


def test_unknown_acuity_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad_acuity.parquet"
    pd.DataFrame(
        {
            "visit_id": [1, 2],
            "arrival": [pd.Timestamp("2024-01-01 10:00"), pd.Timestamp("2024-01-01 11:00")],
            "discharge": [pd.Timestamp("2024-01-01 12:00"), pd.Timestamp("2024-01-01 13:00")],
            "acuity": ["C3", "CX"],
        }
    ).to_parquet(bad)
    with pytest.raises(SchemaError, match="Unknown acuity"):
        load_visits(bad)


def test_drop_invalid_intervals(tmp_path: Path) -> None:
    bad = tmp_path / "bad_interval.parquet"
    pd.DataFrame(
        {
            "visit_id": [1, 2],
            "arrival": [pd.Timestamp("2024-01-01 10:00"), pd.Timestamp("2024-01-01 11:00")],
            "discharge": [pd.Timestamp("2024-01-01 09:00"), pd.Timestamp("2024-01-01 12:00")],
            "acuity": ["C3", "C3"],
        }
    ).to_parquet(bad)
    df = load_visits(bad, drop_invalid=True)
    assert len(df) == 1
    assert df.iloc[0]["visit_id"] == 2


def test_derives_los_hours_when_missing(tmp_path: Path) -> None:
    bad = tmp_path / "no_los.parquet"
    pd.DataFrame(
        {
            "visit_id": [1],
            "arrival": [pd.Timestamp("2024-01-01 10:00")],
            "discharge": [pd.Timestamp("2024-01-01 13:30")],
            "acuity": ["C3"],
        }
    ).to_parquet(bad)
    df = load_visits(bad)
    assert df.iloc[0]["los_hours"] == pytest.approx(3.5)
