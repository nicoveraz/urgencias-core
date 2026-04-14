"""Visit-level ED data loader with schema validation.

Expected schema (parquet columns):

    visit_id     int64             — unique row identifier
    arrival      datetime64[ns]    — time patient physically arrived at ED
    discharge    datetime64[ns]    — time patient physically left ED
    acuity       str ("C1".."C5")  — triage category
    disposition  str (optional)    — alta, hospitalizacion, traslado, fallecido, ...
    los_hours    float (optional)  — derived from (discharge - arrival) when absent

Timestamps may be tz-naive or tz-aware; tz-naive is normalized to no-tz.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = ("visit_id", "arrival", "discharge", "acuity")
OPTIONAL_COLUMNS: tuple[str, ...] = ("disposition", "los_hours")

VALID_ACUITY = {"C1", "C2", "C3", "C4", "C5"}


class SchemaError(ValueError):
    """Raised when the input parquet does not conform to the expected schema."""


def load_visits(
    path: str | Path,
    *,
    drop_invalid: bool = True,
    validate_acuity: bool = True,
) -> pd.DataFrame:
    """Load and validate a visit-level parquet file.

    Parameters
    ----------
    path
        Parquet file path. Must contain all REQUIRED_COLUMNS.
    drop_invalid
        When True (default), drop rows where ``discharge <= arrival`` or where
        required columns are null. When False, raise SchemaError instead.
    validate_acuity
        When True (default), check that all acuity values are in C1..C5.

    Returns
    -------
    pd.DataFrame
        Sorted by arrival, with ``los_hours`` computed when absent, and
        timestamps as tz-naive datetime64[ns].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Visit parquet not found: {path}")

    df = pd.read_parquet(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}. Got {list(df.columns)}")

    df["arrival"] = _to_naive_datetime(df["arrival"], col="arrival")
    df["discharge"] = _to_naive_datetime(df["discharge"], col="discharge")

    if df["visit_id"].isna().any():
        raise SchemaError("visit_id contains null values")

    null_arrivals = df["arrival"].isna()
    null_discharges = df["discharge"].isna()
    if (null_arrivals | null_discharges).any():
        bad = int((null_arrivals | null_discharges).sum())
        if drop_invalid:
            df = df.loc[~(null_arrivals | null_discharges)].copy()
        else:
            raise SchemaError(f"{bad} rows have null arrival or discharge")

    bad_interval = df["discharge"] <= df["arrival"]
    if bad_interval.any():
        n_bad = int(bad_interval.sum())
        if drop_invalid:
            df = df.loc[~bad_interval].copy()
        else:
            raise SchemaError(f"{n_bad} rows have discharge <= arrival")

    if validate_acuity:
        bad_acuity = ~df["acuity"].isin(VALID_ACUITY)
        if bad_acuity.any():
            unknown = sorted(set(df.loc[bad_acuity, "acuity"].unique()))
            raise SchemaError(f"Unknown acuity values: {unknown}. Expected {sorted(VALID_ACUITY)}")

    if "los_hours" not in df.columns:
        df["los_hours"] = (df["discharge"] - df["arrival"]).dt.total_seconds() / 3600.0

    if "disposition" not in df.columns:
        df["disposition"] = pd.NA

    df = df.sort_values("arrival").reset_index(drop=True)
    return df


def _to_naive_datetime(s: pd.Series, *, col: str) -> pd.Series:
    ts = pd.to_datetime(s, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise SchemaError(f"Column {col!r} could not be parsed as datetime")
    return ts
