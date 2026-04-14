from __future__ import annotations

import pandas as pd

from urgencias_core.features.respiratory import RESPIRATORY_COLUMNS, respiratory_features


def test_shape_matches_input() -> None:
    ts = pd.date_range("2024-01-01", periods=42, freq="h")
    feats = respiratory_features(ts)
    assert len(feats) == 42
    assert set(feats.columns) == set(RESPIRATORY_COLUMNS)


def test_stub_returns_zeros() -> None:
    ts = pd.date_range("2024-01-01", periods=5, freq="D")
    feats = respiratory_features(ts)
    assert (feats == 0.0).all().all()
