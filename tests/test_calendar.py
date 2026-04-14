from __future__ import annotations

import pandas as pd

from urgencias_core.features.calendar import CalendarConfig, calendar_features


def test_output_columns() -> None:
    ts = pd.date_range("2024-01-01", periods=10, freq="D")
    feats = calendar_features(ts)
    expected = {
        "year", "month", "day", "dayofweek", "dayofyear",
        "week_of_month", "hour", "is_weekend", "is_holiday",
        "holiday_name", "is_bridge_day", "is_school_holiday",
        "is_semana_musical_frutillar",
    }
    assert expected.issubset(set(feats.columns))
    assert len(feats) == 10


def test_known_holidays_detected() -> None:
    ts = pd.to_datetime(
        [
            "2024-09-18",  # Independence Day — national holiday
            "2024-09-19",  # Glorias del Ejército — national holiday
            "2024-12-25",  # Christmas — national holiday
            "2024-03-15",  # ordinary workday
        ]
    )
    feats = calendar_features(ts)
    assert feats["is_holiday"].tolist() == [True, True, True, False]
    assert "Independencia" in feats["holiday_name"].iloc[0] or feats["holiday_name"].iloc[0]


def test_weekend_detection() -> None:
    ts = pd.to_datetime(["2024-01-06", "2024-01-07", "2024-01-08"])  # Sat, Sun, Mon
    feats = calendar_features(ts)
    assert feats["is_weekend"].tolist() == [True, True, False]


def test_bridge_day_detection() -> None:
    # 2024-06-20 (Thu) is a national holiday (Pueblos Indígenas), 2024-06-21
    # (Fri) is a workday. Taking Fri off produces Thu-Fri-Sat-Sun: a 4-day
    # stretch, so Fri qualifies as a bridge day.
    ts = pd.to_datetime(["2024-06-21", "2024-03-15"])
    feats = calendar_features(ts)
    assert bool(feats["is_bridge_day"].iloc[0]) is True
    assert bool(feats["is_bridge_day"].iloc[1]) is False


def test_school_holiday_windows() -> None:
    ts = pd.to_datetime(
        [
            "2024-07-20",  # winter break
            "2024-12-28",  # summer break
            "2024-02-10",  # summer break
            "2024-05-01",  # ordinary
        ]
    )
    feats = calendar_features(ts)
    assert feats["is_school_holiday"].tolist() == [True, True, True, False]


def test_regional_event_frutillar() -> None:
    ts = pd.to_datetime(["2024-02-05", "2024-06-01"])
    feats = calendar_features(ts)
    assert feats["is_semana_musical_frutillar"].tolist() == [True, False]


def test_custom_regional_event() -> None:
    cfg = CalendarConfig(regional_events={"my_event": ((12, 1), (12, 5))})
    ts = pd.to_datetime(["2024-12-03", "2024-12-10"])
    feats = calendar_features(ts, config=cfg)
    assert "is_my_event" in feats.columns
    assert "is_semana_musical_frutillar" not in feats.columns
    assert feats["is_my_event"].tolist() == [True, False]


def test_week_of_month() -> None:
    ts = pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-29"])
    feats = calendar_features(ts)
    assert feats["week_of_month"].tolist() == [1, 2, 3, 5]
