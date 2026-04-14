"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_ed_visits.parquet"


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    assert FIXTURE_PATH.exists(), (
        f"Synthetic fixture missing at {FIXTURE_PATH}. "
        "Run: uv run python scripts/generate_fixture.py"
    )
    return FIXTURE_PATH


@pytest.fixture(scope="session")
def visits(fixture_path: Path) -> pd.DataFrame:
    from urgencias_core.data.loader import load_visits

    return load_visits(fixture_path)


@pytest.fixture(scope="session")
def visits_to_hourly(visits: pd.DataFrame) -> pd.DataFrame:
    from urgencias_core.data.timeseries import hourly_timeseries

    return hourly_timeseries(visits)
