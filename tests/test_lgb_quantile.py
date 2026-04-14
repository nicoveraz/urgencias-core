from __future__ import annotations

import numpy as np
import pandas as pd

from urgencias_core.models.lgb_quantile import LGBQuantileForecaster
from urgencias_core.models.protocol import HorizonSpec


def _simple_hourly_series(n: int = 24 * 30) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    y = 10 + 5 * np.sin(2 * np.pi * ts.hour / 24) + np.random.default_rng(0).normal(0, 1, n)
    return pd.DataFrame({"timestamp": ts, "y": y})


def test_fit_predict_shape() -> None:
    series = _simple_hourly_series()
    fc = LGBQuantileForecaster(quantiles=(0.5, 0.8, 0.9, 0.95), n_estimators=50)
    fc.fit(series, "y")
    h = HorizonSpec(grain="h", length=24)
    pred = fc.predict(h)
    assert list(pred.columns) == ["timestamp", "q50", "q80", "q90", "q95"]
    assert len(pred) == 24


def test_quantile_monotonicity() -> None:
    series = _simple_hourly_series()
    fc = LGBQuantileForecaster(n_estimators=50)
    fc.fit(series, "y")
    pred = fc.predict(HorizonSpec(grain="h", length=48))
    q_vals = pred[["q50", "q80", "q90", "q95"]].to_numpy()
    assert (np.diff(q_vals, axis=1) >= 0).all()


def test_predict_before_fit_raises() -> None:
    fc = LGBQuantileForecaster()
    try:
        fc.predict(HorizonSpec(grain="h", length=1))
    except RuntimeError:
        return
    raise AssertionError("predict before fit should raise RuntimeError")
