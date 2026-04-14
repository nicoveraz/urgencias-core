"""Forecaster protocol and horizon specification.

The protocol is grain-agnostic: ``grain`` is a pandas frequency alias, so the
same forecaster class can be used at hourly, daily, weekly, or monthly grain.

Prediction DataFrames conform to a standard shape: one ``timestamp`` column
plus one column per requested quantile named ``q{int(q*100)}`` (e.g. ``q50``,
``q80``, ``q90``, ``q95``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class HorizonSpec:
    """Forecast horizon configuration.

    Attributes
    ----------
    grain
        Pandas frequency alias. Common values: ``"h"`` (hourly), ``"D"``
        (daily), ``"W-MON"`` (weekly, week ending Monday), ``"ME"`` (month
        end), ``"MS"`` (month start).
    length
        Number of grain-units to forecast.
    quantiles
        Quantile levels to produce. Must be strictly within (0, 1) and sorted
        ascending. Median (0.5) should be included for point-error metrics.
    start
        Optional explicit first forecast timestamp. When ``None``, forecasters
        should default to ``history_end + one grain-unit``.
    """

    grain: str
    length: int
    quantiles: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)
    start: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError(f"length must be positive, got {self.length}")
        if not self.quantiles:
            raise ValueError("quantiles must be non-empty")
        if any(q <= 0 or q >= 1 for q in self.quantiles):
            raise ValueError(f"quantiles must be in (0, 1), got {self.quantiles}")
        if list(self.quantiles) != sorted(self.quantiles):
            raise ValueError(f"quantiles must be sorted ascending, got {self.quantiles}")

    @property
    def quantile_columns(self) -> tuple[str, ...]:
        """Standard column names for the quantile predictions."""
        return tuple(f"q{int(round(q * 100))}" for q in self.quantiles)


class Forecaster(Protocol):
    """Minimal forecaster interface used by the eval harness.

    Implementations accept a DataFrame with a ``timestamp`` column and one or
    more numeric target columns, and return a DataFrame with ``timestamp`` plus
    the standard quantile columns.
    """

    def fit(self, history: pd.DataFrame, target_col: str) -> None:
        """Fit on an historical series."""

    def predict(self, horizon: HorizonSpec) -> pd.DataFrame:
        """Produce quantile forecasts for the requested horizon."""


def next_timestamp(end: pd.Timestamp, grain: str) -> pd.Timestamp:
    """Return ``end + one grain-unit`` using the pandas offset registry."""
    offset = pd.tseries.frequencies.to_offset(grain)
    return pd.Timestamp(end) + offset


def future_index(
    history_end: pd.Timestamp,
    horizon: HorizonSpec,
) -> pd.DatetimeIndex:
    """Build the DatetimeIndex of forecast timestamps for ``horizon``."""
    start = horizon.start if horizon.start is not None else next_timestamp(history_end, horizon.grain)
    return pd.date_range(start=start, periods=horizon.length, freq=horizon.grain)


__all__ = [
    "Forecaster",
    "HorizonSpec",
    "future_index",
    "next_timestamp",
]
