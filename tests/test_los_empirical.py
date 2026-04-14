from __future__ import annotations

import numpy as np
import pandas as pd

from urgencias_core.simulation.los_empirical import EmpiricalLOSSampler


def test_fit_and_acuity_mix_from_fixture(visits: pd.DataFrame) -> None:
    sampler = EmpiricalLOSSampler(seed=0).fit(visits)
    mix = sampler.acuity_mix
    assert set(mix.keys()).issubset({"C1", "C2", "C3", "C4", "C5"})
    assert abs(sum(mix.values()) - 1.0) < 1e-9


def test_sample_shape_and_positive(visits: pd.DataFrame) -> None:
    sampler = EmpiricalLOSSampler(seed=0).fit(visits)
    s = sampler.sample("C3", hour=10, n=100)
    assert s.shape == (100,)
    assert (s > 0).all()


def test_sample_reproducible_with_seed(visits: pd.DataFrame) -> None:
    a = EmpiricalLOSSampler(seed=7).fit(visits).sample("C3", 8, 50)
    b = EmpiricalLOSSampler(seed=7).fit(visits).sample("C3", 8, 50)
    np.testing.assert_array_equal(a, b)


def test_sample_many_groups_correctly(visits: pd.DataFrame) -> None:
    sampler = EmpiricalLOSSampler(seed=0).fit(visits)
    acuities = np.array(["C3", "C4", "C3", "C5", "C2"])
    hours = np.array([10, 10, 12, 3, 22])
    out = sampler.sample_many(acuities, hours)
    assert out.shape == (5,)
    assert (out > 0).all()


def test_unknown_acuity_falls_back_to_global(visits: pd.DataFrame) -> None:
    sampler = EmpiricalLOSSampler(seed=0).fit(visits)
    s = sampler.sample("ZZ", 0, 100)
    assert s.shape == (100,)
    assert (s > 0).all()


def test_log_normal_fallback_for_thin_bucket() -> None:
    # Build a visits df with a thin bucket (only 5 samples for (C1, 3))
    rng = np.random.default_rng(0)
    rows = []
    for acuity in ["C3", "C4"]:
        for hour in range(24):
            for _ in range(50):
                rows.append((pd.Timestamp("2024-01-01") + pd.Timedelta(hours=hour), acuity, rng.exponential(2)))
    for _ in range(5):
        rows.append((pd.Timestamp("2024-01-01") + pd.Timedelta(hours=3), "C1", rng.exponential(10)))
    df = pd.DataFrame(rows, columns=["arrival", "acuity", "los_hours"])

    sampler = EmpiricalLOSSampler(min_samples=20, seed=0).fit(df)
    # (C1, 3) has 5 observations < min_samples=20 -> log-normal fallback expected.
    s = sampler.sample("C1", 3, n=200)
    assert s.shape == (200,)
    assert (s > 0).all()
