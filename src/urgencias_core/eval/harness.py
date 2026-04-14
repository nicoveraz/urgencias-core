"""Eval harness: train every registered forecaster, hold out the last N
periods, and report MAE/RMSE/MAPE + quantile (pinball) loss per quantile.

Includes a ≥5% improvement rule: the best candidate must beat the best
baseline by at least ``warning_threshold`` on the check quantile (default
P80 pinball loss), otherwise a visible warning is printed to stdout.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from urgencias_core.models.protocol import Forecaster, HorizonSpec


@dataclass
class HarnessReport:
    """Structured result of a harness run."""

    table: pd.DataFrame
    warning_triggered: bool
    warning_message: str


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Pinball loss at quantile ``q``."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def run_harness(
    series: pd.DataFrame,
    target_col: str,
    horizon: HorizonSpec,
    holdout_length: int,
    baselines: dict[str, Forecaster],
    candidates: dict[str, Forecaster] | None = None,
    warning_threshold: float = 0.05,
    check_quantile: float = 0.80,
    verbose: bool = True,
) -> HarnessReport:
    """Run the eval harness.

    Parameters
    ----------
    series
        History with ``timestamp`` column plus ``target_col``. Must be sorted.
    target_col
        Column name to forecast.
    horizon
        Forecast spec. ``length`` is overridden to ``holdout_length`` internally.
    holdout_length
        Number of final periods to hold out as the test set.
    baselines
        Forecasters treated as the baseline bar (SeasonalNaive + statsforecast).
    candidates
        Optional candidate forecasters. The ≥5% rule checks the best candidate
        against the best baseline.
    warning_threshold
        Fractional improvement required of the best candidate, default 0.05.
    check_quantile
        Quantile at which the improvement is checked, default 0.80.
    verbose
        When True, prints the table and any warning to stdout.
    """
    if holdout_length <= 0 or holdout_length >= len(series):
        raise ValueError(
            f"holdout_length must be in (0, len(series)={len(series)}), got {holdout_length}"
        )

    train = series.iloc[:-holdout_length].copy()
    test = series.iloc[-holdout_length:].copy()
    test_start = pd.to_datetime(test["timestamp"].iloc[0])

    eval_horizon = replace(horizon, length=holdout_length, start=test_start)

    rows = []
    candidates = candidates or {}
    all_forecasters = [(n, f, "baseline") for n, f in baselines.items()]
    all_forecasters += [(n, f, "candidate") for n, f in candidates.items()]

    missing_quantiles: dict[str, list[int]] = {}
    for name, fc, role in all_forecasters:
        fc.fit(train, target_col)
        pred = fc.predict(eval_horizon)
        merged = pred.merge(test[["timestamp", target_col]], on="timestamp", how="inner")
        if len(merged) == 0:
            raise RuntimeError(
                f"Forecaster {name!r} produced no timestamps aligned with the holdout window"
            )
        y_true = np.asarray(merged[target_col], dtype="float64")

        if "q50" in merged.columns:
            y50 = np.asarray(merged["q50"], dtype="float64")
            mae = float(np.mean(np.abs(y_true - y50)))
            rmse = float(np.sqrt(np.mean((y_true - y50) ** 2)))
            nonzero = y_true != 0
            mape = (
                float(np.mean(np.abs((y_true[nonzero] - y50[nonzero]) / y_true[nonzero])))
                if nonzero.any()
                else float("nan")
            )
        else:
            mae = rmse = mape = float("nan")

        row: dict = {"name": name, "role": role, "mae": mae, "rmse": rmse, "mape": mape}
        missing_for_this: list[int] = []
        for q in horizon.quantiles:
            q_int = int(round(q * 100))
            col = f"q{q_int}"
            if col not in merged.columns:
                row[f"qloss_{q_int}"] = float("nan")
                missing_for_this.append(q_int)
            else:
                row[f"qloss_{q_int}"] = quantile_loss(
                    y_true, np.asarray(merged[col], dtype="float64"), q
                )
        if missing_for_this:
            missing_quantiles[name] = missing_for_this
        rows.append(row)

    table = pd.DataFrame(rows).set_index("name")

    warning_triggered = False
    warning_message = ""
    check_col = f"qloss_{int(round(check_quantile * 100))}"
    if candidates and check_col in table.columns:
        baseline_table = table[table["role"] == "baseline"].dropna(subset=check_col)
        candidate_table = table[table["role"] == "candidate"].dropna(subset=check_col)
        if len(baseline_table) > 0 and len(candidate_table) > 0:
            best_baseline = baseline_table[check_col].min()
            best_baseline_name = baseline_table[check_col].idxmin()
            best_candidate = candidate_table[check_col].min()
            best_candidate_name = candidate_table[check_col].idxmin()
            target = best_baseline * (1 - warning_threshold)
            if best_candidate > target:
                warning_triggered = True
                warning_message = (
                    f"Best candidate ({best_candidate_name}, {check_col}={best_candidate:.4f}) "
                    f"does not beat best baseline ({best_baseline_name}, {check_col}="
                    f"{best_baseline:.4f}) by >= {warning_threshold:.0%}. "
                    f"A model that does not earn its complexity should not ship."
                )

    if verbose:
        print("\n=== Harness results (lower is better) ===")
        print(table.to_string(float_format=lambda v: f"{v:.4f}"))
        if missing_quantiles:
            print("\nNote: forecasters below did not produce all requested quantiles.")
            print("NaN cells reflect a model-specific capability gap, not a bug.")
            for name, qs in missing_quantiles.items():
                qs_str = ", ".join(f"q{q}" for q in qs)
                print(f"  - {name}: missing {qs_str}")
        if warning_triggered:
            banner = "=" * 78
            print(f"\n{banner}")
            print("WARNING: complexity not justified")
            print(warning_message)
            print(banner)

    return HarnessReport(
        table=table,
        warning_triggered=warning_triggered,
        warning_message=warning_message,
    )


__all__ = ["HarnessReport", "run_harness", "quantile_loss"]
