"""Open-Meteo client for Chilean cities, with on-disk parquet cache.

Puerto Montt is the default location (the Eunosia primary site). Any
coordinates can be supplied. The free Open-Meteo API requires no key.

Two endpoints are supported:

- Historical archive (``archive-api.open-meteo.com``) for training.
- Forecast (``api.open-meteo.com``) for inference.

Results are cached as parquet keyed by (endpoint, coords, dates, variables).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pandas as pd

PUERTO_MONTT = (-41.4689, -72.9411)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DEFAULT_HOURLY_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "relative_humidity_2m",
    "weathercode",
)


@dataclass
class WeatherConfig:
    latitude: float = PUERTO_MONTT[0]
    longitude: float = PUERTO_MONTT[1]
    timezone: str = "America/Santiago"
    variables: tuple[str, ...] = field(default_factory=lambda: DEFAULT_HOURLY_VARIABLES)
    cache_dir: Path = field(default_factory=lambda: Path(".cache/weather"))


def fetch_history(
    start_date: str,
    end_date: str,
    config: WeatherConfig | None = None,
    *,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Fetch hourly weather history from Open-Meteo archive.

    Parameters
    ----------
    start_date, end_date
        ISO date strings ``YYYY-MM-DD``.
    config
        Location + variables + cache settings. Defaults to Puerto Montt.
    client
        Optional httpx.Client for testing (injected mocks).

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp`` plus one column per variable.
    """
    cfg = config or WeatherConfig()
    params = {
        "latitude": cfg.latitude,
        "longitude": cfg.longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(cfg.variables),
        "timezone": cfg.timezone,
    }
    return _cached_fetch(ARCHIVE_URL, params, cfg.cache_dir, client=client)


def fetch_forecast(
    days: int = 7,
    config: WeatherConfig | None = None,
    *,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Fetch hourly forecast from Open-Meteo."""
    cfg = config or WeatherConfig()
    params = {
        "latitude": cfg.latitude,
        "longitude": cfg.longitude,
        "forecast_days": int(days),
        "hourly": ",".join(cfg.variables),
        "timezone": cfg.timezone,
    }
    return _cached_fetch(FORECAST_URL, params, cfg.cache_dir, client=client)


def _cached_fetch(
    url: str,
    params: dict,
    cache_dir: Path,
    *,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = hashlib.sha1(
        (url + json.dumps(params, sort_keys=True, default=str)).encode()
    ).hexdigest()[:16]
    cache_file = cache_dir / f"{key}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()
    finally:
        if owns_client:
            client.close()

    df = _open_meteo_hourly_to_frame(payload)
    df.to_parquet(cache_file, index=False)
    return df


def _open_meteo_hourly_to_frame(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly") or {}
    if "time" not in hourly:
        raise ValueError("Open-Meteo response missing hourly.time")
    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
