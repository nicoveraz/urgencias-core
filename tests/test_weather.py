from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd

from urgencias_core.features.weather import WeatherConfig, fetch_forecast, fetch_history

MOCK_PAYLOAD = {
    "hourly": {
        "time": ["2024-07-01T00:00", "2024-07-01T01:00", "2024-07-01T02:00"],
        "temperature_2m": [8.1, 7.9, 7.5],
        "precipitation": [0.0, 0.3, 0.1],
        "wind_speed_10m": [12.0, 11.5, 10.0],
        "relative_humidity_2m": [85, 88, 90],
        "weathercode": [61, 61, 63],
    }
}


def _mock_client() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=MOCK_PAYLOAD)

    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


def test_fetch_history_parses_payload(tmp_path: Path) -> None:
    cfg = WeatherConfig(cache_dir=tmp_path / "cache")
    with _mock_client() as client:
        df = fetch_history("2024-07-01", "2024-07-01", cfg, client=client)
    assert list(df.columns) == [
        "timestamp",
        "temperature_2m",
        "precipitation",
        "wind_speed_10m",
        "relative_humidity_2m",
        "weathercode",
    ]
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert len(df) == 3


def test_cache_round_trip(tmp_path: Path) -> None:
    cfg = WeatherConfig(cache_dir=tmp_path / "cache")
    with _mock_client() as client:
        first = fetch_history("2024-07-01", "2024-07-01", cfg, client=client)

    # Second call with a failing client should still succeed from cache.
    def fail_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network call made despite cache hit")

    with httpx.Client(transport=httpx.MockTransport(fail_handler)) as client:
        second = fetch_history("2024-07-01", "2024-07-01", cfg, client=client)
    pd.testing.assert_frame_equal(first, second)


def test_fetch_forecast(tmp_path: Path) -> None:
    cfg = WeatherConfig(cache_dir=tmp_path / "cache")
    with _mock_client() as client:
        df = fetch_forecast(days=3, config=cfg, client=client)
    assert len(df) == 3
    assert "temperature_2m" in df.columns


def test_default_coordinates_are_puerto_montt() -> None:
    cfg = WeatherConfig()
    assert round(cfg.latitude, 2) == -41.47
    assert round(cfg.longitude, 2) == -72.94


def _serialize_payload_as_bytes() -> bytes:
    return json.dumps(MOCK_PAYLOAD).encode()
