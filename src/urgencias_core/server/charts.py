"""Matplotlib helpers that return base64-inline PNGs for the reference server."""

from __future__ import annotations

import base64
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _to_base64_png(fig, dpi: int = 110) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def hill_plot(hourly: pd.DataFrame, column: str = "occupancy") -> str:
    """Mean ``column`` by (day-of-week, hour-of-day) — the classic ED hill plot."""
    df = hourly.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["_dow"] = ts.dt.dayofweek
    df["_hour"] = ts.dt.hour
    pivot = df.groupby(["_dow", "_hour"])[column].mean().unstack("_hour")

    dow_names = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for dow, row in pivot.iterrows():
        label = dow_names[dow] if 0 <= dow < 7 else str(dow)
        ax.plot(row.index, row.values, label=label, linewidth=1.3)
    ax.set_xlabel("Hora del día")
    ax.set_ylabel(f"{column} (media)")
    ax.set_title(f"{column.capitalize()} por hora y día de la semana")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(ncol=7, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.grid(True, alpha=0.25)
    return _to_base64_png(fig)


def forecast_chart(pred: pd.DataFrame, target: str) -> str:
    """Forecast chart with uncertainty bands (P50, P80, P90, P95)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    x = pd.to_datetime(pred["timestamp"])
    if {"q80", "q95"}.issubset(pred.columns):
        ax.fill_between(x, pred["q80"], pred["q95"], alpha=0.18, label="P80–P95")
    if {"q50", "q80"}.issubset(pred.columns):
        ax.fill_between(x, pred["q50"], pred["q80"], alpha=0.30, label="P50–P80")
    if "q50" in pred.columns:
        ax.plot(x, pred["q50"], color="#214", linewidth=1.5, label="Mediana")
    ax.set_xlabel("Hora")
    ax.set_ylabel(target)
    ax.set_title(f"Pronóstico — {target}")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    return _to_base64_png(fig)


def fan_chart(sim_quantiles: pd.DataFrame) -> str:
    """Simulation fan chart: per-hour quantile envelopes around the median."""
    fig, ax = plt.subplots(figsize=(9, 4))
    x = sim_quantiles["hour_offset"]
    if {"q80", "q95"}.issubset(sim_quantiles.columns):
        ax.fill_between(x, sim_quantiles["q80"], sim_quantiles["q95"], alpha=0.18, label="P80–P95")
    if {"q50", "q80"}.issubset(sim_quantiles.columns):
        ax.fill_between(x, sim_quantiles["q50"], sim_quantiles["q80"], alpha=0.30, label="P50–P80")
    if "q50" in sim_quantiles.columns:
        ax.plot(x, sim_quantiles["q50"], color="#214", linewidth=1.5, label="Mediana")
    ax.set_xlabel("Horas desde inicio")
    ax.set_ylabel("Censo simulado")
    ax.set_title("Simulación Monte Carlo — distribución de censo")
    ax.legend()
    ax.grid(True, alpha=0.25)
    return _to_base64_png(fig)


def histogram(values: np.ndarray, title: str, xlabel: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(values, bins=40, color="#356", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frecuencia")
    ax.grid(True, alpha=0.25)
    return _to_base64_png(fig)


__all__ = ["fan_chart", "forecast_chart", "hill_plot", "histogram"]
