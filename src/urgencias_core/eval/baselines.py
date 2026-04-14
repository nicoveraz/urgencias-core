"""Baselines the eval harness uses to establish the accuracy bar.

- ``SeasonalNaiveBaseline``: empirical quantiles conditional on a seasonal
  key inferred from the grain (``(dow, hour)`` for hourly, ``(month, dow)``
  for daily, etc.).
- ``StatsForecastWrapper``: thin wrapper around any statsforecast model that
  produces quantile intervals, mapping the ``level`` API back to the standard
  q50/q80/q90/q95 columns.
- Convenience factories for the baseline battery: ``auto_arima``,
  ``auto_ets``, ``auto_theta``, ``mstl``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from urgencias_core.models.protocol import HorizonSpec, future_index


class SeasonalNaiveBaseline:
    """Empirical quantile lookup conditional on a grain-specific seasonal key.

    The lookup key depends on the inferred frequency:
    - Hourly (``h``): ``(dayofweek, hour)``.
    - Daily (``D``): ``(month, dayofweek)``.
    - Weekly (``W*``): ``(isoweek,)``.
    - Monthly (``M*``): ``(month,)``.

    Unknown keys at predict time fall back to the global empirical quantiles.
    """

    def __init__(self, quantiles: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)) -> None:
        self.quantiles = tuple(quantiles)
        self._lookup: dict[tuple, np.ndarray] = {}
        self._fallback: np.ndarray | None = None
        self._key_fn: Callable[[pd.Timestamp], tuple] | None = None
        self._history_end: pd.Timestamp | None = None

    def fit(self, history: pd.DataFrame, target_col: str) -> None:
        ts = pd.to_datetime(history["timestamp"])
        freq = pd.infer_freq(ts)
        self._key_fn = _key_func(freq)
        keys = pd.Series([self._key_fn(t) for t in ts])
        y = np.asarray(history[target_col], dtype="float64")
        df = pd.DataFrame({"key": keys.values, "y": y})
        for k, g in df.groupby("key", sort=False):
            self._lookup[k] = np.quantile(g["y"].to_numpy(), self.quantiles)
        self._fallback = np.quantile(y, self.quantiles)
        self._history_end = ts.max()

    def predict(self, horizon: HorizonSpec) -> pd.DataFrame:
        if self._key_fn is None or self._history_end is None or self._fallback is None:
            raise RuntimeError("SeasonalNaiveBaseline.predict called before fit")
        key_fn = _key_func(horizon.grain) if _grain_changed(self._key_fn, horizon.grain) else self._key_fn
        future = future_index(self._history_end, horizon)
        qcols = [f"q{int(round(q * 100))}" for q in sorted(self.quantiles)]
        sorted_q_idx = np.argsort(self.quantiles)
        values = np.zeros((len(future), len(self.quantiles)), dtype="float64")
        for i, t in enumerate(future):
            k = key_fn(t)
            row = self._lookup.get(k, self._fallback)
            values[i] = row[sorted_q_idx]
        out = pd.DataFrame({"timestamp": future})
        for j, col in enumerate(qcols):
            out[col] = values[:, j]
        return out


class StatsForecastWrapper:
    """Wrap a statsforecast model to conform to the Forecaster protocol.

    Quantile columns are derived from statsforecast's ``level`` API:
    - ``q50`` = point forecast (statsforecast median).
    - ``q80`` = hi-60 interval upper edge.
    - ``q90`` = hi-80 interval upper edge.
    - ``q95`` = hi-90 interval upper edge.
    """

    def __init__(
        self,
        model,
        quantiles: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
        name: str | None = None,
    ) -> None:
        self.model = model
        self.quantiles = tuple(quantiles)
        self.name = name or model.__class__.__name__
        self._sf = None
        self._freq: str | None = None
        self._alias: str | None = None

    def fit(self, history: pd.DataFrame, target_col: str) -> None:
        from statsforecast import StatsForecast

        ts = pd.to_datetime(history["timestamp"])
        freq = pd.infer_freq(ts) or "h"
        df = pd.DataFrame(
            {
                "unique_id": "series",
                "ds": ts.to_numpy(),
                "y": np.asarray(history[target_col], dtype="float64"),
            }
        )
        self._sf = StatsForecast(models=[self.model], freq=freq)
        self._sf.fit(df)
        self._freq = freq

    def predict(self, horizon: HorizonSpec) -> pd.DataFrame:
        if self._sf is None:
            raise RuntimeError(f"{self.name}.predict called before fit")
        levels = sorted({_quantile_to_level(q) for q in self.quantiles if q != 0.5})
        pred = self._sf.predict(h=horizon.length, level=levels)
        if "unique_id" in pred.columns:
            pred = pred.drop(columns=["unique_id"])
        alias = _infer_alias(pred)
        out = pd.DataFrame({"timestamp": pd.to_datetime(pred["ds"]).to_numpy()})
        for q in sorted(self.quantiles):
            col = f"q{int(round(q * 100))}"
            if q == 0.5:
                out[col] = pred[alias].to_numpy()
            else:
                level = _quantile_to_level(q)
                out[col] = pred[f"{alias}-hi-{level}"].to_numpy()
        return out


def auto_arima(
    season_length: int = 24,
    quantiles: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
    **kwargs,
) -> StatsForecastWrapper:
    from statsforecast.models import AutoARIMA

    model = AutoARIMA(season_length=season_length, **kwargs)
    return StatsForecastWrapper(model, quantiles=quantiles, name="AutoARIMA")


def auto_ets(
    season_length: int = 24,
    quantiles: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
    **kwargs,
) -> StatsForecastWrapper:
    from statsforecast.models import AutoETS

    model = AutoETS(season_length=season_length, **kwargs)
    return StatsForecastWrapper(model, quantiles=quantiles, name="AutoETS")


def auto_theta(
    season_length: int = 24,
    quantiles: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
    **kwargs,
) -> StatsForecastWrapper:
    from statsforecast.models import AutoTheta

    model = AutoTheta(season_length=season_length, **kwargs)
    return StatsForecastWrapper(model, quantiles=quantiles, name="AutoTheta")


def mstl(
    season_length: list[int] | tuple[int, ...] = (24, 168),
    quantiles: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
    **kwargs,
) -> StatsForecastWrapper:
    from statsforecast.models import MSTL, AutoARIMA

    trend_fc = kwargs.pop("trend_forecaster", None) or AutoARIMA()
    model = MSTL(season_length=list(season_length), trend_forecaster=trend_fc, **kwargs)
    return StatsForecastWrapper(model, quantiles=quantiles, name="MSTL")


def _quantile_to_level(q: float) -> int:
    """Convert a one-sided upper quantile (>0.5) to a statsforecast interval level."""
    level = int(round((2 * q - 1) * 100))
    return level


def _key_func(freq: str | None) -> Callable[[pd.Timestamp], tuple]:
    if not freq:
        return lambda t: (t.dayofweek, t.hour)
    f = freq.upper()
    if f.startswith("H") or f == "h":
        return lambda t: (t.dayofweek, t.hour)
    if f.startswith("D"):
        return lambda t: (t.month, t.dayofweek)
    if f.startswith("W"):
        return lambda t: (int(t.isocalendar().week),)
    if f.startswith("M"):
        return lambda t: (t.month,)
    return lambda t: (t.dayofweek, t.hour)


def _grain_changed(fitted_fn: Callable, horizon_grain: str) -> bool:
    # Rebuild key function if grain differs. We don't store the original freq
    # string, so safest is to always rebuild from horizon.grain on predict.
    return True


def _infer_alias(pred: pd.DataFrame) -> str:
    for col in pred.columns:
        if col == "ds":
            continue
        if "-lo-" in col or "-hi-" in col:
            continue
        return col
    raise ValueError(f"Could not infer model alias from predict columns: {list(pred.columns)}")


__all__ = [
    "SeasonalNaiveBaseline",
    "StatsForecastWrapper",
    "auto_arima",
    "auto_ets",
    "auto_theta",
    "mstl",
]
