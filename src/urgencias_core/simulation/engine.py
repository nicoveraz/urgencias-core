"""Monte Carlo discrete-event simulator for ED census over a short horizon.

The simulator takes an arrivals forecast (hourly mean arrivals), an empirical
LOS sampler, and the current state of the ED (patients already in care and,
optionally, their hours-in-ED), then runs ``n_sims`` independent Poisson
arrival trajectories. Each arrival samples an acuity from a configurable mix
and a LOS from the sampler; each current patient contributes a residual LOS
drawn from its acuity bucket.

The output is the per-simulation, per-hour census matrix, plus convenience
derivatives: per-hour empirical quantiles and exceedance probabilities for
user-supplied thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .los_empirical import EmpiricalLOSSampler


@dataclass
class CurrentPatient:
    """A patient already in the ED at simulation start (t=0)."""

    acuity: str
    hours_in_ed: float = 0.0


@dataclass
class SimulationResult:
    """Result of a Monte Carlo simulation run.

    Attributes
    ----------
    census
        Array of shape ``(n_sims, horizon_hours)`` with end-of-hour census.
    hours_since_start
        Array of shape ``(horizon_hours,)`` with hour offsets 1..H.
    """

    census: np.ndarray
    hours_since_start: np.ndarray
    quantiles: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)

    def quantile_frame(self) -> pd.DataFrame:
        """Per-hour empirical quantiles across simulations."""
        q = np.quantile(self.census, self.quantiles, axis=0)
        cols = {f"q{int(round(p * 100))}": q[i] for i, p in enumerate(self.quantiles)}
        return pd.DataFrame({"hour_offset": self.hours_since_start, **cols})

    def exceedance(self, threshold: float) -> np.ndarray:
        """Empirical P(census > threshold) for each future hour."""
        return np.mean(self.census > threshold, axis=0)


def simulate(
    arrivals_mean: np.ndarray,
    start_hour: int,
    los_sampler: EmpiricalLOSSampler,
    current_patients: list[CurrentPatient] | int = 0,
    arrival_acuity_mix: dict[str, float] | None = None,
    n_sims: int = 1000,
    quantiles: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
    seed: int = 42,
) -> SimulationResult:
    """Run a Monte Carlo simulation of ED census.

    Parameters
    ----------
    arrivals_mean
        Expected arrivals per hour for each of the next ``H`` hours.
    start_hour
        Hour-of-day at simulation start (0..23). Used to determine which
        LOS bucket each hour's arrivals sample from.
    los_sampler
        A fit ``EmpiricalLOSSampler``.
    current_patients
        Either a list of ``CurrentPatient`` (exact) or an integer count. If
        an integer, patients are drawn from the sampler's empirical acuity
        mix and assumed to have just arrived at t=0.
    arrival_acuity_mix
        Probability per acuity code for future arrivals. Defaults to the
        sampler's empirical mix.
    n_sims
        Number of simulation trajectories (default 1000).
    quantiles
        Quantile levels to summarize the result with.
    seed
        RNG seed for reproducibility.
    """
    arrivals_mean = np.asarray(arrivals_mean, dtype="float64")
    if arrivals_mean.ndim != 1:
        raise ValueError("arrivals_mean must be a 1-D array")
    horizon = len(arrivals_mean)
    if horizon == 0:
        raise ValueError("arrivals_mean must be non-empty")

    mix = arrival_acuity_mix or los_sampler.acuity_mix
    acuity_codes = np.array(list(mix.keys()))
    acuity_probs = np.asarray(list(mix.values()), dtype="float64")
    acuity_probs = acuity_probs / acuity_probs.sum()

    rng = np.random.default_rng(seed)
    census = np.zeros((n_sims, horizon), dtype="int32")

    if isinstance(current_patients, int):
        baseline_count = current_patients
        baseline_acuities_sample = rng.choice(acuity_codes, size=baseline_count, p=acuity_probs)
        baseline_known: list[CurrentPatient] = [
            CurrentPatient(acuity=str(a), hours_in_ed=0.0)
            for a in baseline_acuities_sample
        ]
    else:
        baseline_known = list(current_patients)

    for s in range(n_sims):
        # Residual LOS for current patients: sample fresh from acuity bucket
        # at current hour_of_day. Subtract hours already in ED (truncate to 0).
        depart_times: list[float] = []
        if baseline_known:
            acuities = np.array([p.acuity for p in baseline_known])
            hours = np.full(len(baseline_known), start_hour, dtype="int64")
            los_samples = los_sampler.sample_many(acuities, hours)
            already_in = np.array([p.hours_in_ed for p in baseline_known], dtype="float64")
            remaining = np.maximum(los_samples - already_in, 1 / 60)
            depart_times.extend(remaining.tolist())

        # Per-hour arrivals
        for h in range(horizon):
            n_arrivals = rng.poisson(arrivals_mean[h])
            if n_arrivals > 0:
                arrived_acuities = rng.choice(acuity_codes, size=n_arrivals, p=acuity_probs)
                arrival_hod = (start_hour + h) % 24
                hours = np.full(n_arrivals, arrival_hod, dtype="int64")
                los_new = los_sampler.sample_many(arrived_acuities, hours)
                # Arrivals occur uniformly within the hour; approximate by placing
                # them at hour start. Departure time (in hours since sim start):
                depart_times.extend((h + los_new).tolist())

            # End-of-hour census: count patients with depart_time > (h + 1)
            end_of_hour = h + 1
            alive = sum(1 for d in depart_times if d > end_of_hour)
            census[s, h] = alive

    hours_since_start = np.arange(1, horizon + 1)
    return SimulationResult(
        census=census,
        hours_since_start=hours_since_start,
        quantiles=tuple(quantiles),
    )


__all__ = ["CurrentPatient", "SimulationResult", "simulate"]
