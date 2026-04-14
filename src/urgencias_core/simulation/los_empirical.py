"""Empirical length-of-stay sampler conditional on (acuity, hour_of_arrival).

At fit time, stores a dense empirical quantile grid per (acuity, hour) bucket.
Sampling is done via inverse-CDF transform: draw u ~ Uniform(0, 1), return
``interp(u, quantile_grid, bucket_quantiles)``.

Thin buckets (fewer than ``min_samples`` observations) fall back to a
log-normal fit to the bucket's log-mean and log-sigma. Completely unseen
keys fall back to the global empirical distribution.

This sampler conflates clinical workup time with inpatient boarding time —
the historical dataset does not distinguish them. Forecasts that use this
sampler therefore implicitly assume future boarding patterns resemble the
training period. Documented in ``docs/decisions.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EmpiricalLOSSampler:
    """Inverse-CDF LOS sampler with log-normal smoothing for thin buckets."""

    quantile_grid: np.ndarray = field(
        default_factory=lambda: np.linspace(0.005, 0.995, 200)
    )
    min_samples: int = 20
    min_los_hours: float = 1 / 60  # 1 minute floor
    seed: int = 42

    _rng: np.random.Generator = field(init=False, repr=False)
    _empirical: dict = field(init=False, default_factory=dict)
    _lognormal: dict = field(init=False, default_factory=dict)
    _fallback_quantiles: np.ndarray | None = field(init=False, default=None)
    _acuity_mix: dict[str, float] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def fit(self, visits: pd.DataFrame) -> "EmpiricalLOSSampler":
        """Fit on a visit-level DataFrame with ``arrival``, ``acuity``, ``los_hours``."""
        required = {"arrival", "acuity", "los_hours"}
        missing = required - set(visits.columns)
        if missing:
            raise ValueError(f"visits missing columns: {sorted(missing)}")

        hour = visits["arrival"].dt.hour.to_numpy()
        acuity = visits["acuity"].to_numpy()
        los = np.clip(visits["los_hours"].to_numpy(dtype="float64"), self.min_los_hours, None)

        frame = pd.DataFrame({"acuity": acuity, "hour": hour, "los": los})
        for (a, h), g in frame.groupby(["acuity", "hour"], sort=False):
            samples = g["los"].to_numpy()
            if len(samples) >= self.min_samples:
                self._empirical[(a, int(h))] = np.quantile(samples, self.quantile_grid)
            elif len(samples) >= 2:
                log_samples = np.log(samples)
                self._lognormal[(a, int(h))] = (float(log_samples.mean()), float(log_samples.std(ddof=0)))

        self._fallback_quantiles = np.quantile(los, self.quantile_grid)

        acuity_counts = frame["acuity"].value_counts(normalize=True)
        self._acuity_mix = {str(k): float(v) for k, v in acuity_counts.items()}

        return self

    @property
    def acuity_mix(self) -> dict[str, float]:
        """Empirical acuity distribution observed at fit time."""
        if not self._acuity_mix:
            raise RuntimeError("Sampler has not been fit yet")
        return dict(self._acuity_mix)

    def sample(self, acuity: str, hour: int, n: int = 1) -> np.ndarray:
        """Draw ``n`` LOS samples for a given (acuity, hour) bucket."""
        if self._fallback_quantiles is None:
            raise RuntimeError("Sampler has not been fit yet")
        key = (acuity, int(hour))
        if key in self._empirical:
            u = self._rng.uniform(size=n)
            return np.interp(u, self.quantile_grid, self._empirical[key])
        if key in self._lognormal:
            mu, sigma = self._lognormal[key]
            sigma = max(sigma, 1e-6)
            return self._rng.lognormal(mean=mu, sigma=sigma, size=n)
        u = self._rng.uniform(size=n)
        return np.interp(u, self.quantile_grid, self._fallback_quantiles)

    def sample_many(self, acuities: np.ndarray, hours: np.ndarray) -> np.ndarray:
        """Batch-sample LOS for parallel arrays of ``(acuity, hour)`` pairs."""
        acuities = np.asarray(acuities)
        hours = np.asarray(hours, dtype="int64")
        n = len(acuities)
        if len(hours) != n:
            raise ValueError("acuities and hours must have the same length")
        out = np.empty(n, dtype="float64")
        if n == 0:
            return out
        frame = pd.DataFrame({"a": acuities, "h": hours})
        for (a, h), idx in frame.groupby(["a", "h"], sort=False).indices.items():
            out[idx] = self.sample(str(a), int(h), n=len(idx))
        return out


__all__ = ["EmpiricalLOSSampler"]
