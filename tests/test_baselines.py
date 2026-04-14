from __future__ import annotations

import numpy as np
import pandas as pd

from urgencias_core.eval.baselines import (
    SeasonalNaiveBaseline,
    auto_arima,
    auto_ets,
    auto_theta,
)
from urgencias_core.models.protocol import HorizonSpec


def test_seasonal_naive_returns_expected_quantiles() -> None:
    # Build a deterministic hourly series where (dow=0, hour=0) always equals 10
    # and every other (dow, hour) always equals 1.
    n_weeks = 8
    n = 24 * 7 * n_weeks
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    y = np.where((ts.dayofweek == 0) & (ts.hour == 0), 10.0, 1.0)
    series = pd.DataFrame({"timestamp": ts, "y": y})

    fc = SeasonalNaiveBaseline(quantiles=(0.5, 0.8, 0.9, 0.95))
    fc.fit(series, "y")

    h = HorizonSpec(grain="h", length=24 * 7)
    pred = fc.predict(h)
    # Series ends 2024-02-25 23:00 (Sun). First forecast is 2024-02-26 00:00 (Mon),
    # which is a (dow=0, hour=0) key — always 10.0 in training.
    first_ts = pd.Timestamp("2024-02-26 00:00")
    mon_00 = pred[pred["timestamp"] == first_ts]
    assert float(mon_00["q50"].iloc[0]) == 10.0
    # (dow=0, hour=1) was always 1 — all quantiles = 1.
    mon_01 = pred[pred["timestamp"] == pd.Timestamp("2024-02-26 01:00")]
    assert float(mon_01["q50"].iloc[0]) == 1.0
    assert float(mon_01["q95"].iloc[0]) == 1.0


def test_statsforecast_auto_arima_smoke() -> None:
    n = 24 * 14
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    y = 5 + np.sin(2 * np.pi * ts.hour / 24) + np.random.default_rng(0).normal(0, 0.2, n)
    series = pd.DataFrame({"timestamp": ts, "y": y})

    fc = auto_arima(season_length=24)
    fc.fit(series, "y")
    pred = fc.predict(HorizonSpec(grain="h", length=12))
    assert list(pred.columns) == ["timestamp", "q50", "q80", "q90", "q95"]
    assert len(pred) == 12


def test_statsforecast_auto_ets_and_theta_smoke() -> None:
    n = 24 * 10
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    y = np.sin(2 * np.pi * ts.hour / 24) + 5.0
    series = pd.DataFrame({"timestamp": ts, "y": y})

    for factory in (auto_ets, auto_theta):
        fc = factory(season_length=24)
        fc.fit(series, "y")
        pred = fc.predict(HorizonSpec(grain="h", length=6))
        assert len(pred) == 6
        assert {"q50", "q80", "q90", "q95"}.issubset(pred.columns)
