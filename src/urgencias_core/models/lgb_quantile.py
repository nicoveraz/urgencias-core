"""Generic LightGBM quantile regression forecaster.

One LGBMRegressor per requested quantile with ``objective='quantile'``.
Features are calendar-only (hour/day/month plus sin-cos encodings). This
means the model is purely seasonal: it cannot extrapolate a trend and it
does not use autoregressive lags. The benefit is that it is grain-agnostic
and honest about information availability at forecast time.

Hospitals wanting AR-lagged variants for short horizons should subclass or
reimplement: downstream ``eunosia-forecast`` models do exactly this for the
Andes Salud deployment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .protocol import HorizonSpec, future_index


class LGBQuantileForecaster:
    """LightGBM quantile regression forecaster (calendar features only).

    Conforms to the ``Forecaster`` protocol.
    """

    FEATURE_COLS: tuple[str, ...] = (
        "hour", "dow", "month", "doy", "week_of_month", "year", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    )

    def __init__(
        self,
        quantiles: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_data_in_leaf: int = 20,
        seed: int = 42,
    ) -> None:
        self.quantiles = tuple(quantiles)
        self._base_params: dict = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_data_in_leaf": min_data_in_leaf,
            "verbose": -1,
            "seed": seed,
        }
        self._models: dict[float, LGBMRegressor] = {}
        self._history_end: pd.Timestamp | None = None

    def fit(self, history: pd.DataFrame, target_col: str) -> None:
        ts = pd.to_datetime(history["timestamp"])
        X = self._make_features(ts)
        y = np.asarray(history[target_col], dtype="float64")
        for q in self.quantiles:
            params = {**self._base_params, "objective": "quantile", "alpha": float(q)}
            model = LGBMRegressor(**params)
            model.fit(X, y)
            self._models[q] = model
        self._history_end = ts.max()

    def predict(self, horizon: HorizonSpec) -> pd.DataFrame:
        if self._history_end is None:
            raise RuntimeError("LGBQuantileForecaster.predict called before fit")
        future = future_index(self._history_end, horizon)
        X = self._make_features(pd.Series(future))
        out = pd.DataFrame({"timestamp": future})
        for q in sorted(self.quantiles):
            col = f"q{int(round(q * 100))}"
            out[col] = self._models[q].predict(X)
        # Enforce quantile monotonicity across rows (repair any crossings).
        qcols = [f"q{int(round(q * 100))}" for q in sorted(self.quantiles)]
        out[qcols] = np.sort(out[qcols].to_numpy(), axis=1)
        return out

    @classmethod
    def _make_features(cls, ts: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
        s = ts if isinstance(ts, pd.Series) else pd.Series(ts)
        s = pd.to_datetime(s)
        hour = s.dt.hour.to_numpy(dtype="int16")
        dow = s.dt.dayofweek.to_numpy(dtype="int16")
        month = s.dt.month.to_numpy(dtype="int16")
        doy = s.dt.dayofyear.to_numpy(dtype="int16")
        day = s.dt.day.to_numpy(dtype="int16")
        week_of_month = (((day - 1) // 7) + 1).astype("int16")
        year = s.dt.year.to_numpy(dtype="int16")
        is_weekend = (dow >= 5).astype("int16")

        return pd.DataFrame(
            {
                "hour": hour,
                "dow": dow,
                "month": month,
                "doy": doy,
                "week_of_month": week_of_month,
                "year": year,
                "is_weekend": is_weekend,
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
                "dow_sin": np.sin(2 * np.pi * dow / 7),
                "dow_cos": np.cos(2 * np.pi * dow / 7),
                "month_sin": np.sin(2 * np.pi * month / 12),
                "month_cos": np.cos(2 * np.pi * month / 12),
            }
        )


__all__ = ["LGBQuantileForecaster"]
