"""Respiratory virus activity features — STUB.

The Instituto de Salud Pública (ISP) publishes weekly respiratory bulletins
(ESSAR / Vigilancia Laboratorial Respiratoria) containing VRS, influenza, and
other positive rates by epidemiological week. Ingesting these produces a
strong seasonal signal for ED pediatric and respiratory-related arrivals.

This stub returns zeros with the expected shape so that downstream models can
depend on the feature without blocking on the ingestion. Replace the body with
a real ingestion when ISP bulletins are parsed into a tabular source.

TODO: ingest ISP ESSAR bulletins.
    - Source: https://boletin.ispch.gob.cl/ (HTML + PDF)
    - Weekly grain, published Tuesdays, 1-2 week lag
    - Output columns: respiratory_activity (0-1 index), vrs_positive_rate,
      influenza_positive_rate, sars_cov2_positive_rate
"""

from __future__ import annotations

import pandas as pd

RESPIRATORY_COLUMNS: tuple[str, ...] = (
    "respiratory_activity",
    "vrs_positive_rate",
    "influenza_positive_rate",
    "sars_cov2_positive_rate",
)


def respiratory_features(
    timestamps: pd.Series | pd.DatetimeIndex,
) -> pd.DataFrame:
    """Return a DataFrame of respiratory features aligned to input timestamps.

    Currently a stub: all values are 0.0. Shape and column names are stable so
    that downstream models can rely on them.
    """
    n = len(timestamps)
    return pd.DataFrame({col: [0.0] * n for col in RESPIRATORY_COLUMNS})
