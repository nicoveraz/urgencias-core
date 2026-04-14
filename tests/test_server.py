from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from urgencias_core.server.app import create_app
from urgencias_core.server.config import DataConfig, ServerConfig, SimulationConfig


def _client(fixture_path: Path) -> TestClient:
    cfg = ServerConfig(
        data=DataConfig(parquet=fixture_path),
        simulation=SimulationConfig(horizon_hours=12, n_sims=50, current_census=5, start_hour=8),
    )
    app = create_app(cfg)
    return TestClient(app)


def test_index_route(fixture_path: Path) -> None:
    with _client(fixture_path) as client:
        r = client.get("/")
    assert r.status_code == 200
    assert "urgencias-core" in r.text
    assert "Análisis descriptivo" in r.text


def test_baseline_route(fixture_path: Path) -> None:
    with _client(fixture_path) as client:
        r = client.get("/baseline")
    assert r.status_code == 200
    assert "data:image/png;base64" in r.text
    assert "agudeza" in r.text.lower()


def test_forecast_route_with_defaults(fixture_path: Path) -> None:
    with _client(fixture_path) as client:
        r = client.get("/forecast", params={"horizon": 24})
    assert r.status_code == 200
    assert "q50" in r.text
    assert "data:image/png;base64" in r.text


def test_forecast_route_lgb(fixture_path: Path) -> None:
    with _client(fixture_path) as client:
        r = client.get("/forecast", params={"horizon": 24, "forecaster": "lgb_quantile"})
    assert r.status_code == 200
    assert "lgb_quantile" in r.text


def test_simulation_route(fixture_path: Path) -> None:
    with _client(fixture_path) as client:
        r = client.get(
            "/simulation",
            params={"horizon": 12, "n_sims": 50, "current_census": 8, "start_hour": 14, "arrivals": 6.0},
        )
    assert r.status_code == 200
    assert "Monte Carlo" in r.text
    assert "data:image/png;base64" in r.text


def test_static_css_served(fixture_path: Path) -> None:
    with _client(fixture_path) as client:
        r = client.get("/static/style.css")
    assert r.status_code == 200
    assert "data-table" in r.text
