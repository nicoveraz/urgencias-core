"""TOML config loader for the reference HTTP server.

The server reads a single TOML file (default: ``urgencias-core.toml`` in the
current working directory). When no file is present it falls back to a
built-in default that points at the packaged synthetic fixture so the server
runs out of the box on a fresh clone.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_FIXTURE = _PACKAGE_ROOT / "tests" / "fixtures" / "synthetic_ed_visits.parquet"


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parquet: Path = Field(
        default=_DEFAULT_FIXTURE,
        description="Path to a visit-level parquet file matching the loader schema.",
    )


class ForecastConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forecaster: str = Field(
        default="seasonal_naive",
        description="Forecaster to run on /forecast. One of: seasonal_naive, lgb_quantile.",
    )
    target: str = Field(
        default="arrivals",
        description="Column from the hourly time series to forecast.",
    )
    horizon_hours: int = Field(default=168, ge=1, le=24 * 30)


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    horizon_hours: int = Field(default=24, ge=1, le=72)
    n_sims: int = Field(default=200, ge=10, le=5000)
    current_census: int = Field(default=0, ge=0)
    start_hour: int = Field(default=12, ge=0, le=23)
    arrivals_per_hour: float | None = Field(
        default=None,
        description=(
            "Flat arrivals mean per hour. When None, uses the recent history "
            "mean from the hourly time series."
        ),
    )


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig = Field(default_factory=DataConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)


DEFAULT_CONFIG_FILENAME = "urgencias-core.toml"


def load_config(path: str | Path | None = None) -> ServerConfig:
    """Load and validate a TOML config. Falls back to defaults when absent."""
    if path is None:
        candidate = Path.cwd() / DEFAULT_CONFIG_FILENAME
        if not candidate.exists():
            return ServerConfig()
        path = candidate
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return ServerConfig(**raw)


__all__ = [
    "DataConfig",
    "ForecastConfig",
    "ServerConfig",
    "SimulationConfig",
    "load_config",
    "DEFAULT_CONFIG_FILENAME",
]
