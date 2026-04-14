from __future__ import annotations

import numpy as np
import pandas as pd

from urgencias_core.simulation.engine import CurrentPatient, simulate
from urgencias_core.simulation.los_empirical import EmpiricalLOSSampler


def _fit_sampler(visits: pd.DataFrame) -> EmpiricalLOSSampler:
    return EmpiricalLOSSampler(seed=0).fit(visits)


def test_zero_arrivals_and_empty_ed_gives_zero_census(visits: pd.DataFrame) -> None:
    sampler = _fit_sampler(visits)
    result = simulate(
        arrivals_mean=np.zeros(24),
        start_hour=0,
        los_sampler=sampler,
        current_patients=0,
        n_sims=50,
        seed=1,
    )
    assert result.census.shape == (50, 24)
    assert (result.census == 0).all()


def test_current_patients_drain_over_time(visits: pd.DataFrame) -> None:
    sampler = _fit_sampler(visits)
    start_patients = [CurrentPatient(acuity="C3") for _ in range(50)]
    result = simulate(
        arrivals_mean=np.zeros(48),
        start_hour=10,
        los_sampler=sampler,
        current_patients=start_patients,
        n_sims=100,
        seed=1,
    )
    mean_by_hour = result.census.mean(axis=0)
    # Census should be monotonically non-increasing on average with zero arrivals.
    assert mean_by_hour[0] <= len(start_patients)
    # By hour 48 nearly everyone should have departed (median LOS < 3h).
    assert mean_by_hour[-1] < mean_by_hour[0] * 0.2


def test_exceedance_probabilities_bounded(visits: pd.DataFrame) -> None:
    sampler = _fit_sampler(visits)
    result = simulate(
        arrivals_mean=np.full(24, 5.0),
        start_hour=12,
        los_sampler=sampler,
        current_patients=10,
        n_sims=200,
        seed=2,
    )
    p = result.exceedance(threshold=5)
    assert p.shape == (24,)
    assert ((p >= 0.0) & (p <= 1.0)).all()


def test_reproducible_with_seed(visits: pd.DataFrame) -> None:
    sampler = _fit_sampler(visits)
    args = dict(
        arrivals_mean=np.full(12, 4.0),
        start_hour=8,
        los_sampler=sampler,
        current_patients=5,
        n_sims=50,
        seed=42,
    )
    r1 = simulate(**args)
    # Rebuild sampler so its internal RNG state is also fresh.
    args["los_sampler"] = _fit_sampler(visits)
    r2 = simulate(**args)
    np.testing.assert_array_equal(r1.census, r2.census)


def test_quantile_frame_has_expected_shape(visits: pd.DataFrame) -> None:
    sampler = _fit_sampler(visits)
    result = simulate(
        arrivals_mean=np.full(6, 3.0),
        start_hour=0,
        los_sampler=sampler,
        current_patients=0,
        n_sims=100,
        seed=3,
    )
    qf = result.quantile_frame()
    assert list(qf.columns) == ["hour_offset", "q50", "q80", "q90", "q95"]
    assert len(qf) == 6
    assert (qf["q50"] <= qf["q95"]).all()
