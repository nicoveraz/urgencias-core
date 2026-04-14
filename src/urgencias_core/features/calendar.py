"""Chilean calendar features for ED forecasting.

Combines pyholidays.CL, a hardcoded approximate school calendar, bridge-day
logic, and configurable regional events. The default regional event is Semana
Musical de Frutillar (early February), documented as an example for other
hospitals to replace with their own local events.

Sources
-------
- National holidays: ``holidays.CL`` (pyholidays package).
- School calendar: Chilean Ministry of Education (MINEDUC) annual resolutions.
  Dates vary year-to-year; this module uses conservative approximations that
  capture the bulk of each break. Hospitals wanting exact dates should subclass
  ``CalendarConfig`` with year-specific overrides.
- Semana Musical de Frutillar: https://semanamusicaldefrutillar.cl (first
  ~10 days of February each year; exact dates vary).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import holidays as pyholidays
import pandas as pd

DEFAULT_REGIONAL_EVENTS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    # name -> ((start_month, start_day), (end_month, end_day))
    "semana_musical_frutillar": ((2, 1), (2, 10)),
}


@dataclass
class CalendarConfig:
    """Configuration for calendar feature generation.

    Attributes
    ----------
    country
        ISO country code for pyholidays. Defaults to ``"CL"``.
    regional_events
        Mapping of event name -> ((start_month, start_day), (end_month, end_day)).
        Each event becomes a ``is_<name>`` boolean column. Override to add or
        remove events for your hospital's region.
    """

    country: str = "CL"
    regional_events: dict[str, tuple[tuple[int, int], tuple[int, int]]] = field(
        default_factory=lambda: dict(DEFAULT_REGIONAL_EVENTS)
    )


def calendar_features(
    timestamps: pd.Series | pd.DatetimeIndex,
    config: CalendarConfig | None = None,
) -> pd.DataFrame:
    """Generate calendar features for each timestamp.

    Returns a DataFrame aligned to the input with columns:

        year, month, day, dayofweek, dayofyear, week_of_month, hour
        is_weekend            bool
        is_holiday            bool
        holiday_name          str ("" when not a holiday)
        is_bridge_day         bool — workday between a holiday and a weekend
        is_school_holiday     bool — summer break, winter break, Fiestas Patrias
        is_<event_name>       bool for each configured regional event
    """
    cfg = config or CalendarConfig()

    if isinstance(timestamps, pd.Series):
        ts = pd.to_datetime(timestamps).reset_index(drop=True)
    else:
        ts = pd.Series(pd.DatetimeIndex(timestamps))

    years = sorted(ts.dt.year.unique().tolist())
    cl_holidays = pyholidays.country_holidays(cfg.country, years=years)

    out = pd.DataFrame(index=ts.index)
    out["year"] = ts.dt.year.astype("int16")
    out["month"] = ts.dt.month.astype("int8")
    out["day"] = ts.dt.day.astype("int8")
    out["dayofweek"] = ts.dt.dayofweek.astype("int8")
    out["dayofyear"] = ts.dt.dayofyear.astype("int16")
    out["week_of_month"] = (((ts.dt.day - 1) // 7) + 1).astype("int8")
    out["hour"] = ts.dt.hour.astype("int8")
    out["is_weekend"] = out["dayofweek"].isin([5, 6])

    dates_only = ts.dt.date
    out["is_holiday"] = dates_only.map(lambda d: d in cl_holidays)
    out["holiday_name"] = dates_only.map(lambda d: cl_holidays.get(d, ""))

    out["is_bridge_day"] = dates_only.map(lambda d: _is_bridge_day(d, cl_holidays))

    out["is_school_holiday"] = dates_only.map(_is_school_holiday)

    for name, ((m0, d0), (m1, d1)) in cfg.regional_events.items():
        out[f"is_{name}"] = _in_annual_window(ts, m0, d0, m1, d1)

    return out


def _is_bridge_day(d: date, holidays_lookup: pyholidays.HolidayBase) -> bool:
    """A bridge day (``puente``) is a weekday workday adjacent to a holiday
    such that taking the day off would produce a 3+ day stretch of consecutive
    non-workdays (counting holidays + weekends on either side).

    Examples (Chile):
    - Tue after Mon holiday: Sat-Sun-MonHol-TueOff → bridge.
    - Fri after Thu holiday: ThuHol-FriOff-Sat-Sun → bridge.
    - Mon before Tue holiday: Sat-Sun-MonOff-TueHol → bridge.
    """
    if d in holidays_lookup:
        return False
    if d.weekday() >= 5:
        return False

    def _is_off(day: date) -> bool:
        return day in holidays_lookup or day.weekday() >= 5

    left = 0
    cur = d - timedelta(days=1)
    while _is_off(cur):
        left += 1
        cur -= timedelta(days=1)
    right = 0
    cur = d + timedelta(days=1)
    while _is_off(cur):
        right += 1
        cur += timedelta(days=1)

    adjacent_to_holiday = (
        (d - timedelta(days=1)) in holidays_lookup
        or (d + timedelta(days=1)) in holidays_lookup
    )
    return adjacent_to_holiday and (left + 1 + right) >= 3


def _is_school_holiday(d: date) -> bool:
    """Approximate Chilean school calendar breaks.

    - Summer break: Dec 20 - Feb 28
    - Winter break: Jul 12 - Jul 26 (middle two weeks)
    - Fiestas Patrias break: Sep 17 - Sep 20

    Dates are approximate; MINEDUC varies year to year. Override via a custom
    CalendarConfig if your hospital needs exact dates.
    """
    md = (d.month, d.day)
    if md >= (12, 20) or md <= (2, 28):
        return True
    if (7, 12) <= md <= (7, 26):
        return True
    if (9, 17) <= md <= (9, 20):
        return True
    return False


def _in_annual_window(
    ts: pd.Series,
    m_start: int,
    d_start: int,
    m_end: int,
    d_end: int,
) -> pd.Series:
    month_day = list(zip(ts.dt.month, ts.dt.day, strict=True))
    start = (m_start, d_start)
    end = (m_end, d_end)
    if start <= end:
        return pd.Series([start <= md <= end for md in month_day], index=ts.index)
    # wraps year end (e.g. Dec -> Feb)
    return pd.Series([md >= start or md <= end for md in month_day], index=ts.index)
