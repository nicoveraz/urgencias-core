"""Minimal FastAPI reference server.

Server-rendered HTML with embedded base64 PNG charts. Zero JS, no build step.
Designed to be launched with::

    uv run uvicorn urgencias_core.server.app:app

It loads a single parquet (configured via ``urgencias-core.toml``, default:
the packaged synthetic fixture) at startup and serves three views:

- ``/baseline``  — descriptive analytics over the full history
- ``/forecast``  — runs the configured forecaster on the recent window
- ``/simulation``— runs the Monte Carlo engine with configurable parameters
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from urgencias_core.data.loader import load_visits
from urgencias_core.data.timeseries import hourly_timeseries
from urgencias_core.eval.baselines import SeasonalNaiveBaseline
from urgencias_core.models.lgb_quantile import LGBQuantileForecaster
from urgencias_core.models.protocol import HorizonSpec
from urgencias_core.simulation.engine import simulate
from urgencias_core.simulation.los_empirical import EmpiricalLOSSampler

from . import charts
from .config import ServerConfig, load_config

BASE_DIR = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@dataclass
class ServerState:
    """In-memory state built once at startup."""

    config: ServerConfig
    visits: pd.DataFrame
    hourly: pd.DataFrame
    los_sampler: EmpiricalLOSSampler


def _build_state(config: ServerConfig) -> ServerState:
    visits = load_visits(config.data.parquet)
    hourly = hourly_timeseries(visits)
    sampler = EmpiricalLOSSampler(seed=0).fit(visits)
    return ServerState(config=config, visits=visits, hourly=hourly, los_sampler=sampler)


def _select_forecaster(name: str):
    name = name.lower().strip()
    if name in ("seasonal_naive", "naive"):
        return SeasonalNaiveBaseline()
    if name in ("lgb_quantile", "lgb"):
        return LGBQuantileForecaster(n_estimators=200)
    raise ValueError(f"Unknown forecaster: {name!r}")


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Build the FastAPI app."""
    cfg = config or load_config()
    state = _build_state(cfg)

    app = FastAPI(title="urgencias-core reference server", version="0")
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    app.state.server_state = state

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        ctx = {
            "n_visits": len(state.visits),
            "n_hours": len(state.hourly),
            "history_start": state.visits["arrival"].min(),
            "history_end": state.visits["arrival"].max(),
            "data_path": str(cfg.data.parquet),
        }
        return TEMPLATES.TemplateResponse(request, "index.html", ctx)

    @app.get("/baseline", response_class=HTMLResponse)
    def baseline(request: Request):
        occ_chart = charts.hill_plot(state.hourly, column="occupancy")
        arr_chart = charts.hill_plot(state.hourly, column="arrivals")

        los_summary = (
            state.visits.groupby("acuity")["los_hours"]
            .agg(
                n="count",
                median="median",
                p80=lambda s: float(s.quantile(0.80)),
                p95=lambda s: float(s.quantile(0.95)),
            )
            .round(2)
        )
        los_table = los_summary.to_html(classes="data-table", border=0)

        overall = {
            "total_visits": int(len(state.visits)),
            "median_los": float(state.visits["los_hours"].median()),
            "mean_occupancy": float(state.hourly["occupancy"].mean()),
            "peak_hour": int(
                state.hourly.groupby(pd.to_datetime(state.hourly["timestamp"]).dt.hour)[
                    "occupancy"
                ]
                .mean()
                .idxmax()
            ),
        }

        ctx = {
            "occupancy_chart": occ_chart,
            "arrivals_chart": arr_chart,
            "los_table": los_table,
            "overall": overall,
        }
        return TEMPLATES.TemplateResponse(request, "baseline.html", ctx)

    @app.get("/forecast", response_class=HTMLResponse)
    def forecast(
        request: Request,
        horizon: int = Query(default=None, ge=1, le=24 * 30),
        forecaster: str = Query(default=None),
        target: str = Query(default=None),
    ):
        fc_name = forecaster or cfg.forecast.forecaster
        tgt = target or cfg.forecast.target
        h_hours = int(horizon or cfg.forecast.horizon_hours)

        fc = _select_forecaster(fc_name)
        # Use a recent window to keep reference forecasts quick.
        train = state.hourly.iloc[-24 * 90 :]
        fc.fit(train, tgt)
        pred = fc.predict(HorizonSpec(grain="h", length=h_hours))

        chart = charts.forecast_chart(pred, target=tgt)
        table_df = pred.head(48).copy()
        numeric_cols = [c for c in table_df.columns if c != "timestamp"]
        table_df[numeric_cols] = table_df[numeric_cols].round(2)
        table = table_df.to_html(classes="data-table", border=0, index=False)

        ctx = {
            "forecaster": fc_name,
            "target": tgt,
            "horizon_hours": h_hours,
            "chart": chart,
            "table": table,
            "rows_shown": min(48, len(pred)),
            "rows_total": len(pred),
        }
        return TEMPLATES.TemplateResponse(request, "forecast.html", ctx)

    @app.get("/simulation", response_class=HTMLResponse)
    def simulation(
        request: Request,
        horizon: int = Query(default=None, ge=1, le=72),
        n_sims: int = Query(default=None, ge=10, le=5000),
        current_census: int = Query(default=None, ge=0, le=500),
        start_hour: int = Query(default=None, ge=0, le=23),
        arrivals: float = Query(default=None, ge=0.0, le=200.0),
    ):
        h = int(horizon or cfg.simulation.horizon_hours)
        n = int(n_sims or cfg.simulation.n_sims)
        census0 = int(current_census if current_census is not None else cfg.simulation.current_census)
        sh = int(start_hour if start_hour is not None else cfg.simulation.start_hour)

        if arrivals is None:
            arrivals_effective = cfg.simulation.arrivals_per_hour
            if arrivals_effective is None:
                arrivals_effective = float(state.hourly["arrivals"].tail(24 * 30).mean())
        else:
            arrivals_effective = float(arrivals)

        arrivals_mean = np.full(h, arrivals_effective, dtype="float64")

        result = simulate(
            arrivals_mean=arrivals_mean,
            start_hour=sh,
            los_sampler=state.los_sampler,
            current_patients=census0,
            n_sims=n,
            seed=1,
        )
        qf = result.quantile_frame()
        chart = charts.fan_chart(qf)
        table = qf.round(1).to_html(classes="data-table", border=0, index=False)

        exceedance_thresholds = [int(q) for q in np.linspace(max(1, census0), max(1, census0) + 10, 3)]
        exceedance_rows = []
        for t in exceedance_thresholds:
            probs = result.exceedance(t)
            exceedance_rows.append({"threshold": t, "max_prob": float(probs.max())})

        ctx = {
            "horizon": h,
            "n_sims": n,
            "current_census": census0,
            "start_hour": sh,
            "arrivals_per_hour": round(arrivals_effective, 2),
            "chart": chart,
            "table": table,
            "exceedance_rows": exceedance_rows,
        }
        return TEMPLATES.TemplateResponse(request, "simulation.html", ctx)

    return app


app = create_app()


__all__ = ["app", "create_app", "ServerState"]
