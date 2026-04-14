"""DEIS demo: forecasting-only on real Chilean public hospital data.

Runs against DEIS MINSAL *Atenciones de Urgencia* open data for Hospital de
Puerto Montt (24-105) and Hospital de Frutillar (24-115). If the live fetch
fails or the current user is offline, falls back to the frozen snapshot at
``tests/fixtures/deis_demo_snapshot.parquet``.

This demo does NOT invoke the simulation engine — DEIS data is aggregated
counts per facility per day per cause, without the visit-level
arrival/discharge timestamps the simulator needs.

Outputs (under ``outputs/``):

- ``deis_baseline_comparison.md``         baselines × hospitals, holdout metrics
- ``deis_holdout_<hospital>.png``         actual vs. best forecaster on holdout
- ``deis_forecast_<hospital>.png``        live 6-month forward forecast
- ``deis_forecast_<hospital>.csv``        forward forecast as a table
- ``deis_demo_summary.md``                one-page summary

**Methodological framing only.** This demo uses public data to show how the
forecasting layer behaves on real Chilean hospital data. It does NOT
constitute an operational, clinical, or quality evaluation of either
hospital.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from urgencias_core.data.deis import DEMO_HOSPITALS, fetch_demo_hospitals
from urgencias_core.eval.baselines import (
    SeasonalNaiveBaseline,
    auto_arima,
    auto_ets,
    auto_theta,
    mstl,
)
from urgencias_core.eval.harness import quantile_loss, run_harness
from urgencias_core.models.protocol import HorizonSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO_ROOT / "tests" / "fixtures" / "deis_demo_snapshot.parquet"
OUTPUTS = REPO_ROOT / "outputs"

COVID_EXCLUDE_YEARS = {2020, 2021}
HOLDOUT_WEEKS = 12
FORWARD_WEEKS = 26  # ~6 months
TOTAL_CAUSE_GLOSA = "SECCIÓN 1. TOTAL ATENCIONES DE URGENCIA"

DEMO_HOSPITAL_CODE_MAP = {
    # map internal key to the code actually present in the DEIS snapshot
    "hospital_base_puerto_montt": "24-105",
    "hospital_frutillar": "24-115",
}


def _daily_totals(df: pd.DataFrame, facility_code: str) -> pd.DataFrame:
    sub = df[df["facility_code"] == facility_code]
    totals = sub[sub["cause_group"] == TOTAL_CAUSE_GLOSA]
    if len(totals) == 0:
        totals = sub[sub["cause_group"].str.upper().str.contains("SECCI.?N 1", regex=True, na=False)]
    daily = (
        totals.groupby("date")["count"]
        .sum()
        .reset_index()
        .rename(columns={"date": "timestamp"})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    daily["timestamp"] = pd.to_datetime(daily["timestamp"])
    return daily


def _to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    ts = daily.set_index("timestamp").resample("W-MON")["count"].sum().reset_index()
    # Drop the final partial week if it does not cover a full week.
    if len(ts) > 0:
        last_day_expected = ts["timestamp"].iloc[-1]
        data_end = daily["timestamp"].max()
        if data_end < last_day_expected:
            ts = ts.iloc[:-1]
    return ts


def _fit_and_predict(fc, train: pd.DataFrame, target: str, horizon: HorizonSpec) -> pd.DataFrame:
    fc.fit(train, target)
    return fc.predict(horizon)


def _plot_holdout(train: pd.DataFrame, test: pd.DataFrame, pred: pd.DataFrame, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    window = train.tail(52)
    ax.plot(window["timestamp"], window["count"], color="#777", label="Entrenamiento (52 sem)")
    ax.plot(test["timestamp"], test["count"], color="#c33", linewidth=2, label="Real (holdout)")
    ax.fill_between(pred["timestamp"], pred["q80"], pred["q95"], alpha=0.15, label="P80–P95")
    ax.fill_between(pred["timestamp"], pred["q50"], pred["q80"], alpha=0.28, label="P50–P80")
    ax.plot(pred["timestamp"], pred["q50"], color="#214", linewidth=1.6, label="Predicción mediana")
    ax.set_title(title)
    ax.set_xlabel("Semana")
    ax.set_ylabel("Atenciones semanales")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_forward(history: pd.DataFrame, forecast: pd.DataFrame, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    hist = history.tail(104)
    ax.plot(hist["timestamp"], hist["count"], color="#444", label="Historia (2 años)")
    ax.fill_between(forecast["timestamp"], forecast["q80"], forecast["q95"], alpha=0.15, label="P80–P95")
    ax.fill_between(forecast["timestamp"], forecast["q50"], forecast["q80"], alpha=0.28, label="P50–P80")
    ax.plot(forecast["timestamp"], forecast["q50"], color="#214", linewidth=1.8, label="Predicción mediana")
    ax.axvline(hist["timestamp"].iloc[-1], color="#c33", linestyle="--", alpha=0.7, linewidth=0.9)
    ax.set_title(title)
    ax.set_xlabel("Semana")
    ax.set_ylabel("Atenciones semanales")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _fresh_baselines() -> dict:
    """Return a new dict of forecaster instances (statsforecast models retain
    fit state per instance, so we build fresh ones per fit)."""
    return {
        "SeasonalNaive": SeasonalNaiveBaseline(),
        "AutoARIMA": auto_arima(season_length=52),
        "AutoETS": auto_ets(season_length=52),
        "AutoTheta": auto_theta(season_length=52),
        "MSTL": mstl(season_length=[52]),
    }


def run(offline: bool = False, start_year: int = 2022) -> None:
    OUTPUTS.mkdir(exist_ok=True)
    print(f"Loading DEIS data ({'offline' if offline else 'live + fallback'})...")
    if offline:
        df = pd.read_parquet(SNAPSHOT)
    else:
        df = fetch_demo_hospitals(start_year=start_year, snapshot_path=SNAPSHOT)

    df = df[~df["year"].astype(int).isin(COVID_EXCLUDE_YEARS)].copy()
    df["date"] = pd.to_datetime(df["date"])
    print(f"  {len(df):,} rows, {df['date'].min().date()} → {df['date'].max().date()}")

    baseline_md: list[str] = [
        "# Comparación de baselines DEIS — holdout de 12 semanas",
        "",
        f"- Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- Rango de datos: {df['date'].min().date()} a {df['date'].max().date()}",
        f"- Años incluidos: {sorted(df['year'].astype(int).unique().tolist())}",
        f"- Variable objetivo: `{TOTAL_CAUSE_GLOSA}` (suma diaria → agregación semanal)",
        "",
    ]

    summary_rows: list[dict] = []

    for hosp_key, deis_code in DEMO_HOSPITAL_CODE_MAP.items():
        facility_rows = df[df["facility_code"] == deis_code]
        if len(facility_rows) == 0:
            print(f"  [skip] {hosp_key}: no rows for code {deis_code}")
            continue
        facility_name = facility_rows["facility_name"].iloc[0]
        print(f"\n=== {facility_name} ({deis_code}) ===")

        daily = _daily_totals(df, deis_code)
        weekly = _to_weekly(daily)
        if len(weekly) < HOLDOUT_WEEKS + 52:
            print(f"  [skip] not enough weekly observations: {len(weekly)}")
            continue

        train = weekly.iloc[:-HOLDOUT_WEEKS].reset_index(drop=True)
        test = weekly.iloc[-HOLDOUT_WEEKS:].reset_index(drop=True)
        horizon = HorizonSpec(grain="W-MON", length=HOLDOUT_WEEKS)

        harness_baselines = _fresh_baselines()
        report = run_harness(
            series=weekly,
            target_col="count",
            horizon=horizon,
            holdout_length=HOLDOUT_WEEKS,
            baselines=harness_baselines,
            verbose=False,
        )

        best_name = report.table["qloss_80"].idxmin()
        best_mae = report.table.loc[best_name, "mae"]
        best_qloss = report.table.loc[best_name, "qloss_80"]

        baseline_md.append(f"## {facility_name} ({deis_code})")
        baseline_md.append("")
        baseline_md.append(report.table.round(2).to_markdown())
        baseline_md.append("")
        baseline_md.append(f"**Ganador (qloss_80):** `{best_name}`  —  MAE {best_mae:.1f}, qloss80 {best_qloss:.2f}")
        baseline_md.append("")

        # Holdout plot: refit the best model on train, predict the holdout window.
        plot_fc = _fresh_baselines()[best_name]
        plot_pred = _fit_and_predict(plot_fc, train, "count", horizon)
        merged = plot_pred.merge(test, on="timestamp", how="inner")
        if len(merged) < len(test):
            # Pandas/statsforecast sometimes offsets weekly timestamps slightly.
            plot_pred = plot_pred.copy()
            plot_pred["timestamp"] = test["timestamp"].values[: len(plot_pred)]
            merged = plot_pred.merge(test, on="timestamp", how="inner")

        holdout_png = OUTPUTS / f"deis_holdout_{hosp_key}.png"
        _plot_holdout(train, test, plot_pred, f"{facility_name} — holdout 12 semanas ({best_name})", holdout_png)

        # Live forward: refit best on ALL data and forecast FORWARD_WEEKS weeks.
        forward_fc = _fresh_baselines()[best_name]
        forward_horizon = HorizonSpec(grain="W-MON", length=FORWARD_WEEKS)
        forward = _fit_and_predict(forward_fc, weekly, "count", forward_horizon)
        forward_png = OUTPUTS / f"deis_forecast_{hosp_key}.png"
        _plot_forward(weekly, forward, f"{facility_name} — pronóstico semanal 6 meses ({best_name})", forward_png)

        forward_csv = OUTPUTS / f"deis_forecast_{hosp_key}.csv"
        forward[["timestamp", "q50", "q80", "q90"]].to_csv(forward_csv, index=False)

        summary_rows.append(
            {
                "hospital": facility_name,
                "code": deis_code,
                "weeks_available": len(weekly),
                "holdout_start": test["timestamp"].iloc[0].date(),
                "holdout_end": test["timestamp"].iloc[-1].date(),
                "best_baseline": best_name,
                "mae_holdout": best_mae,
                "qloss80_holdout": best_qloss,
                "forecast_start": forward["timestamp"].iloc[0].date(),
                "forecast_end": forward["timestamp"].iloc[-1].date(),
                "forecast_next4w_p50_total": float(forward["q50"].iloc[:4].sum()),
            }
        )

        print(f"  Best: {best_name}  MAE={best_mae:.1f}  qloss80={best_qloss:.2f}")
        print(f"  Wrote {holdout_png.name}, {forward_png.name}, {forward_csv.name}")

    comparison_md_path = OUTPUTS / "deis_baseline_comparison.md"
    comparison_md_path.write_text("\n".join(baseline_md))

    summary_md = [
        "# DEIS Demo — Resumen",
        "",
        "**Enmarcamiento estricto:** esta demostración usa datos públicos de DEIS "
        "MINSAL para mostrar el funcionamiento de las herramientas de forecasting "
        "sobre datos reales de hospitales chilenos. No constituye una evaluación "
        "operacional, clínica ni de calidad de los hospitales mencionados.",
        "",
        f"- Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- Fuente: DEIS MINSAL, *Atenciones de Urgencia* (open data)",
        f"- Rango histórico: {df['date'].min().date()} a {df['date'].max().date()}",
        f"- Años excluidos por política COVID: {sorted(COVID_EXCLUDE_YEARS)}",
        f"- Holdout: últimas {HOLDOUT_WEEKS} semanas completas",
        f"- Horizonte de pronóstico hacia adelante: {FORWARD_WEEKS} semanas (~6 meses)",
        "",
        "## Por hospital",
        "",
    ]
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_md.append(summary_df.round(2).to_markdown(index=False))
    else:
        summary_md.append("_No hubo hospitales con datos suficientes para forecasting._")
    summary_md_path = OUTPUTS / "deis_demo_summary.md"
    summary_md_path.write_text("\n".join(summary_md))

    print("\n=== Resumen ===")
    if summary_rows:
        print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nArtefactos escritos a {OUTPUTS.relative_to(REPO_ROOT)}/:")
    for name in (
        comparison_md_path.name,
        summary_md_path.name,
        *(f"deis_holdout_{k}.png" for k in DEMO_HOSPITAL_CODE_MAP),
        *(f"deis_forecast_{k}.png" for k in DEMO_HOSPITAL_CODE_MAP),
        *(f"deis_forecast_{k}.csv" for k in DEMO_HOSPITAL_CODE_MAP),
    ):
        print(f"  - {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use the snapshot only, no network")
    parser.add_argument("--start-year", type=int, default=2022, help="First DEIS year to fetch (live mode)")
    args = parser.parse_args()
    run(offline=args.offline, start_year=args.start_year)


if __name__ == "__main__":
    main()
