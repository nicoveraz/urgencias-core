"""Visit-level → hourly time series transformation.

The core trick: emit +1 events at each arrival and -1 events at each discharge,
sort by timestamp, cumulative-sum to get instantaneous census, then resample to
hourly. This yields real occupancy (not a proxy) because the input discharge
field reflects physical departure time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ACUITY_LEVELS: tuple[str, ...] = ("C1", "C2", "C3", "C4", "C5")


def hourly_timeseries(
    visits: pd.DataFrame,
    *,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compute hourly time series from a visit-level DataFrame.

    Columns in output:

        timestamp              — hourly datetime index (tz-naive)
        arrivals               — count of arrivals starting in [t, t+1h)
        departures             — count of discharges in [t, t+1h)
        occupancy              — mean instantaneous census during the hour
        occupancy_c1..c5       — per-acuity mean census during the hour
        mean_los_arriving      — mean LOS (h) of visits arriving in this hour
                                 (NaN when arrivals == 0)

    Parameters
    ----------
    visits
        DataFrame from ``loader.load_visits``. Must have columns
        ``arrival``, ``discharge``, ``acuity``, ``los_hours``.
    cache_path
        Optional parquet path. If provided and the file exists, the cached
        timeseries is returned and computation is skipped. If it does not
        exist, the result is written to this path after computation.

    Returns
    -------
    pd.DataFrame
        Hourly time series, sorted by timestamp, ``timestamp`` column.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return pd.read_parquet(cache_path)

    required = {"arrival", "discharge", "acuity", "los_hours"}
    missing = required - set(visits.columns)
    if missing:
        raise ValueError(f"visits DataFrame is missing columns: {sorted(missing)}")

    start = visits["arrival"].min().floor("h")
    end = visits["discharge"].max().ceil("h")
    index = pd.date_range(start=start, end=end, freq="h", name="timestamp")

    occupancy_total = _occupancy_by_events(visits["arrival"], visits["discharge"], index)

    per_acuity = {}
    for level in ACUITY_LEVELS:
        mask = visits["acuity"] == level
        per_acuity[f"occupancy_{level.lower()}"] = _occupancy_by_events(
            visits.loc[mask, "arrival"],
            visits.loc[mask, "discharge"],
            index,
        )

    arr_bin = visits["arrival"].dt.floor("h")
    dis_bin = visits["discharge"].dt.floor("h")
    arrivals = arr_bin.value_counts().reindex(index, fill_value=0).astype("int64")
    departures = dis_bin.value_counts().reindex(index, fill_value=0).astype("int64")
    mean_los_arriving = (
        visits.groupby(arr_bin)["los_hours"].mean().reindex(index).astype("float64")
    )

    out = pd.DataFrame(
        {
            "arrivals": arrivals.to_numpy(),
            "departures": departures.to_numpy(),
            "occupancy": occupancy_total,
            **per_acuity,
            "mean_los_arriving": mean_los_arriving.to_numpy(),
        },
        index=index,
    ).reset_index()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path, index=False)

    return out


def _occupancy_by_events(
    arrivals: pd.Series,
    discharges: pd.Series,
    index: pd.DatetimeIndex,
) -> np.ndarray:
    """Compute mean census over each hour in ``index`` using event accumulation.

    Returns a numpy array aligned to ``index`` with the time-weighted mean of
    the instantaneous census during each hour. The census at any instant is the
    cumulative sum of (+1 at arrival, -1 at discharge) events up to that instant.
    """
    if len(arrivals) == 0:
        return np.zeros(len(index), dtype="float64")

    events = pd.concat(
        [
            pd.DataFrame({"t": arrivals.to_numpy(), "delta": 1}),
            pd.DataFrame({"t": discharges.to_numpy(), "delta": -1}),
        ],
        ignore_index=True,
    )
    events["t"] = pd.to_datetime(events["t"])
    events = events.sort_values("t", kind="mergesort").reset_index(drop=True)
    events["census"] = events["delta"].cumsum().astype("int64")

    times = events["t"].to_numpy()
    census = events["census"].to_numpy()

    hour_ns = np.int64(3_600_000_000_000)
    edges = index.to_numpy().astype("datetime64[ns]").astype("int64")
    edges_end = edges + hour_ns
    times_ns = times.astype("datetime64[ns]").astype("int64")

    out = np.zeros(len(index), dtype="float64")
    for i in range(len(index)):
        t0 = edges[i]
        t1 = edges_end[i]
        left = np.searchsorted(times_ns, t0, side="left")
        right = np.searchsorted(times_ns, t1, side="left")

        base_census = census[left - 1] if left > 0 else 0
        if right == left:
            out[i] = float(base_census)
            continue

        segment_times = np.concatenate(([t0], times_ns[left:right], [t1]))
        segment_census = np.concatenate(([base_census], census[left:right]))
        weights = np.diff(segment_times).astype("float64")
        out[i] = float(np.sum(segment_census * weights) / hour_ns)

    return out
