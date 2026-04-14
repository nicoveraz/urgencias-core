"""The 30-second synthetic demo.

Loads the packaged synthetic fixture, computes the hourly occupancy time
series, fits ``SeasonalNaiveBaseline`` on arrivals, runs the Monte Carlo
simulator for 24 hours, and writes three PNG charts plus a short stdout
summary.

Run with::

    uv run python scripts/demo_synthetic.py

Outputs land in ``outputs/`` at the repo root:

- ``demo_synthetic_occupancy.png``  historical hourly occupancy (last week)
- ``demo_synthetic_forecast.png``   48-hour arrivals forecast with bands
- ``demo_synthetic_simulation.png`` 24-hour Monte Carlo census fan chart
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from urgencias_core.data.loader import load_visits
from urgencias_core.data.timeseries import hourly_timeseries
from urgencias_core.eval.baselines import SeasonalNaiveBaseline
from urgencias_core.models.protocol import HorizonSpec
from urgencias_core.simulation.engine import simulate
from urgencias_core.simulation.los_empirical import EmpiricalLOSSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "synthetic_ed_visits.parquet"
OUTPUTS = REPO_ROOT / "outputs"


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    print(f"Loading fixture: {FIXTURE}")
    visits = load_visits(FIXTURE)
    print(f"  {len(visits):,} visits, {visits['arrival'].min().date()} -> {visits['arrival'].max().date()}")

    print("Computing hourly time series...")
    hourly = hourly_timeseries(visits)
    print(f"  {len(hourly):,} hours")

    # 1. Occupancy chart (last 7 days)
    last_week = hourly.tail(24 * 7)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(last_week["timestamp"], last_week["occupancy"], color="#214", linewidth=1.2)
    ax.set_title("Ocupación horaria — última semana de la historia sintética")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Censo")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.savefig(OUTPUTS / "demo_synthetic_occupancy.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 2. Forecast: fit SeasonalNaive on the last 30 days, predict 48 h ahead
    train = hourly.tail(24 * 30).reset_index(drop=True)
    fc = SeasonalNaiveBaseline()
    fc.fit(train, "arrivals")
    horizon = HorizonSpec(grain="h", length=48)
    pred = fc.predict(horizon)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(pred["timestamp"], pred["q80"], pred["q95"], alpha=0.18, label="P80–P95")
    ax.fill_between(pred["timestamp"], pred["q50"], pred["q80"], alpha=0.30, label="P50–P80")
    ax.plot(pred["timestamp"], pred["q50"], color="#214", linewidth=1.5, label="Mediana")
    ax.set_title("Pronóstico horario de llegadas (SeasonalNaiveBaseline, 48 h)")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Llegadas / hora")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.savefig(OUTPUTS / "demo_synthetic_forecast.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 3. Simulation
    print("Fitting LOS sampler...")
    sampler = EmpiricalLOSSampler(seed=0).fit(visits)
    print("Running Monte Carlo simulation (N=500, 24h forward)...")
    mean_arrivals = float(hourly["arrivals"].tail(24 * 14).mean())
    result = simulate(
        arrivals_mean=np.full(24, mean_arrivals),
        start_hour=12,
        los_sampler=sampler,
        current_patients=int(hourly["occupancy"].iloc[-1]),
        n_sims=500,
        seed=1,
    )
    qf = result.quantile_frame()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    x = qf["hour_offset"]
    ax.fill_between(x, qf["q80"], qf["q95"], alpha=0.18, label="P80–P95")
    ax.fill_between(x, qf["q50"], qf["q80"], alpha=0.30, label="P50–P80")
    ax.plot(x, qf["q50"], color="#214", linewidth=1.5, label="Mediana")
    ax.set_title("Simulación Monte Carlo — censo simulado 24 h")
    ax.set_xlabel("Horas desde inicio")
    ax.set_ylabel("Censo")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.savefig(OUTPUTS / "demo_synthetic_simulation.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Summary
    print("\n=== Resumen ===")
    print(f"Visitas:                    {len(visits):,}")
    print(f"Horas de historia:          {len(hourly):,}")
    print(f"Ocupación media:            {hourly['occupancy'].mean():.2f}")
    print(f"Llegadas/hora (últimas 2s): {mean_arrivals:.2f}")
    print(f"Mediana forecast P50 (48h): {pred['q50'].median():.2f}")
    print(f"Mediana sim P50 (hora 12):  {qf['q50'].iloc[11]:.2f}")
    print(f"\nArtefactos en {OUTPUTS.relative_to(REPO_ROOT)}/:")
    for name in ("demo_synthetic_occupancy.png", "demo_synthetic_forecast.png", "demo_synthetic_simulation.png"):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
