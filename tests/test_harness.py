from __future__ import annotations

import numpy as np
import pandas as pd

from urgencias_core.eval.baselines import SeasonalNaiveBaseline
from urgencias_core.eval.harness import quantile_loss, run_harness
from urgencias_core.models.lgb_quantile import LGBQuantileForecaster
from urgencias_core.models.protocol import HorizonSpec


def test_quantile_loss_known_case() -> None:
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([8.0, 12.0])
    # q=0.5 loss = mean of |y-ŷ|/2 = (2 + 2) / 2 / 2 = 1.0
    assert quantile_loss(y_true, y_pred, 0.5) == 1.0


def test_harness_structure_and_warning_off() -> None:
    n = 24 * 14
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    # Strongly seasonal so seasonal-naive is hard to beat; no candidate model registered.
    y = 10 + 3 * np.sin(2 * np.pi * ts.hour / 24)
    series = pd.DataFrame({"timestamp": ts, "y": y})

    h = HorizonSpec(grain="h", length=24)
    report = run_harness(
        series=series,
        target_col="y",
        horizon=h,
        holdout_length=24,
        baselines={"SeasonalNaive": SeasonalNaiveBaseline()},
        verbose=False,
    )
    assert report.warning_triggered is False
    assert "qloss_50" in report.table.columns
    assert report.table.loc["SeasonalNaive", "role"] == "baseline"


def test_harness_flags_weak_candidate() -> None:
    n = 24 * 14
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    y = 10 + 3 * np.sin(2 * np.pi * ts.hour / 24)
    series = pd.DataFrame({"timestamp": ts, "y": y})

    class ConstantForecaster:
        def fit(self, history: pd.DataFrame, target_col: str) -> None:
            self._mean = float(np.mean(history[target_col]))
            self._end = pd.to_datetime(history["timestamp"]).max()

        def predict(self, horizon: HorizonSpec) -> pd.DataFrame:
            from urgencias_core.models.protocol import future_index

            idx = future_index(self._end, horizon)
            return pd.DataFrame(
                {
                    "timestamp": idx,
                    "q50": self._mean,
                    "q80": self._mean,
                    "q90": self._mean,
                    "q95": self._mean,
                }
            )

    h = HorizonSpec(grain="h", length=24)
    report = run_harness(
        series=series,
        target_col="y",
        horizon=h,
        holdout_length=24,
        baselines={"SeasonalNaive": SeasonalNaiveBaseline()},
        candidates={"Constant": ConstantForecaster()},
        verbose=False,
    )
    assert report.warning_triggered is True
    assert "Constant" in report.warning_message


def test_lgb_candidate_runs(visits_to_hourly: pd.DataFrame) -> None:
    # Minimal end-to-end check with real fixture + LGB.
    sample = visits_to_hourly.iloc[-24 * 60 :].reset_index(drop=True)
    report = run_harness(
        series=sample,
        target_col="arrivals",
        horizon=HorizonSpec(grain="h", length=24),
        holdout_length=24,
        baselines={"SeasonalNaive": SeasonalNaiveBaseline()},
        candidates={"LGBQuantile": LGBQuantileForecaster(n_estimators=80)},
        verbose=False,
    )
    assert {"SeasonalNaive", "LGBQuantile"}.issubset(report.table.index)
