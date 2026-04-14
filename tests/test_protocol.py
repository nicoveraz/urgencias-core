from __future__ import annotations

import pandas as pd
import pytest

from urgencias_core.models.protocol import HorizonSpec, future_index, next_timestamp


def test_horizon_spec_defaults() -> None:
    h = HorizonSpec(grain="h", length=24)
    assert h.quantiles == (0.5, 0.8, 0.9, 0.95)
    assert h.quantile_columns == ("q50", "q80", "q90", "q95")


def test_horizon_spec_validates_quantiles() -> None:
    with pytest.raises(ValueError):
        HorizonSpec(grain="h", length=24, quantiles=(0.8, 0.5))
    with pytest.raises(ValueError):
        HorizonSpec(grain="h", length=24, quantiles=(0.0, 0.5))
    with pytest.raises(ValueError):
        HorizonSpec(grain="h", length=0)


def test_next_timestamp_hourly() -> None:
    t = pd.Timestamp("2024-01-01 10:00")
    assert next_timestamp(t, "h") == pd.Timestamp("2024-01-01 11:00")


def test_next_timestamp_daily() -> None:
    t = pd.Timestamp("2024-01-01")
    assert next_timestamp(t, "D") == pd.Timestamp("2024-01-02")


def test_future_index_length() -> None:
    h = HorizonSpec(grain="h", length=5)
    idx = future_index(pd.Timestamp("2024-01-01 10:00"), h)
    assert len(idx) == 5
    assert idx[0] == pd.Timestamp("2024-01-01 11:00")
