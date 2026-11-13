"""Date and date-time position scales (``scale_x_date``, ``scale_x_datetime`` ...).

A date scale is a :class:`~ggmanim.scales.continuous.ScaleContinuousPosition` whose
transformed space is a plain number: **days** since 1970-01-01 for date scales,
**seconds** since 1970-01-01 for date-time scales. :meth:`ScaleDatePosition.transform`
accepts ``datetime64`` columns/arrays, ``datetime.date``/``datetime.datetime``
objects, pandas ``Timestamp``\\ s, ISO strings (``"2020-03-01"``) and numbers
(already days/seconds since the epoch), so data, ``limits=``, ``breaks=`` and
``coord_cartesian(xlim=)`` can all be given as dates.

Breaks
------
* automatic (default): :func:`date_breaks_auto` picks a calendar unit and step from
  the span -- seconds, minutes, hours, days, weeks, months (1/2/3/6) or years
  (1/2/5/10 ...) -- aligned to the calendar (months to January, weeks to Monday);
* ``date_breaks="1 month"`` / ``"2 weeks"`` / ``"6 hours"`` ... (``"N unit"``,
  ggplot2 style) or a pandas offset alias (``"MS"``, ``"QS"``, ``"YS"``, ``"7D"``);
* ``breaks=`` a sequence of dates, or a callable ``f((lo, hi))`` on numeric limits.

Labels
------
``date_labels="%b %Y"`` (strftime) or ``labels=`` (e.g. :func:`ggmanim.labels.label_date`).
By default the format follows the breaks' resolution: ``"%Y"`` for January-1st
breaks, ``"%b"``/``"%b %Y"`` for month starts, ``"%b %d"``/``"%Y-%m-%d"`` for days,
``"%H:%M"`` for minutes, ``"%H:%M:%S"`` otherwise.

Time zones
----------
The transformed space is always UTC: timezone-aware values are converted to UTC,
naive ones are taken to be UTC already. Date-time scales take ``timezone=`` (an IANA
name such as ``"America/New_York"``, via the stdlib :mod:`zoneinfo`): breaks are
then placed on that zone's calendar (local midnights, month starts ...) and labels
show local time. Without it, the zone of tz-aware data is used when all such
columns share one (as ggplot2 does), else UTC. Pure numpy/pandas; no manim imports.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .._tz import local_to_utc, resolve_tz, tz_name, utc_to_local
from ..registry import register
from .continuous import X_AESTHETICS, Y_AESTHETICS, ScaleContinuousPosition, Transform

__all__ = [
    "DateTransform",
    "ScaleDatePosition",
    "to_epoch",
    "date_kind",
    "data_timezone",
    "parse_date_width",
    "date_breaks_auto",
    "date_breaks_width",
    "date_format_auto",
    "scale_x_date",
    "scale_y_date",
    "scale_x_datetime",
    "scale_y_datetime",
]

_SECONDS = {"D": 86400.0, "s": 1.0}
_DAY = 86400
_EPOCH = np.datetime64("1970-01-01T00:00:00", "us")


# --------------------------------------------------------------------------- conversion
def _is_number(v: Any) -> bool:
    return v is None or (
        isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
    )


def to_epoch(values: Any, unit: str = "D") -> np.ndarray:
    """Dates/date-times -> float ``unit``\\ s since 1970-01-01 (``"D"`` days or ``"s"``
    seconds). Numbers pass through unchanged (they are taken to be in ``unit``
    already); missing values (``None``, ``NaT``, ``NaN``) become NaN. Timezone-aware
    values are converted to UTC."""
    if unit not in _SECONDS:
        raise ValueError(f"unit must be 'D' or 's', got {unit!r}")
    if isinstance(values, (pd.Series, pd.Index)) and isinstance(values.dtype, pd.DatetimeTZDtype):
        values = pd.Series(values).dt.tz_convert(None)
    if isinstance(values, (pd.Timestamp, _dt.date, np.datetime64, str)):
        values = [values]
    arr = np.asarray(values)
    if arr.dtype.kind in "biuf":
        return np.atleast_1d(arr.astype(float)).ravel()
    if arr.dtype.kind == "O" and all(_is_number(v) for v in arr.ravel()):
        return np.atleast_1d(np.array([np.nan if v is None else v for v in arr.ravel()], float))
    if arr.dtype.kind != "M":
        try:
            conv = pd.to_datetime(pd.Series(arr.ravel(), dtype=object), utc=None)
        except (TypeError, ValueError):
            try:  # mixed naive/aware values
                conv = pd.to_datetime(pd.Series(arr.ravel(), dtype=object), utc=True)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"date scale got values that are not dates: {arr.ravel()[:3].tolist()!r}"
                ) from exc
        if isinstance(conv.dtype, pd.DatetimeTZDtype):
            conv = conv.dt.tz_convert(None)
        arr = conv.to_numpy(dtype="datetime64[ns]")
    arr = np.atleast_1d(arr).ravel()
    us = arr.astype("datetime64[us]")
    nat = np.isnat(us)
    out = (us - _EPOCH).astype("int64").astype(float) / 1e6 / _SECONDS[unit]
    out[nat] = np.nan
    return out


def _seconds_to_dt64(sec: np.ndarray) -> np.ndarray:
    return _EPOCH + np.round(np.asarray(sec, dtype=float) * 1e6).astype("int64").astype(
        "timedelta64[us]"
    )


def date_kind(columns: Sequence[pd.Series]) -> str | None:
    """``"datetime"``, ``"date"`` or ``None`` for a set of columns mapped to one axis:
    a date scale is used when some column is ``datetime64`` (naive or tz-aware) or
    holds only ``date``/``datetime``/``Timestamp`` objects. It is a date-time scale
    when some value has a time of day other than midnight, else a date scale."""
    kind = None
    for s in columns:
        s = s.dropna()
        if not len(s):
            continue
        if pd.api.types.is_datetime64_any_dtype(s.dtype):
            vals = pd.Series(s)
        elif s.dtype == object and all(isinstance(v, (_dt.date, np.datetime64)) for v in s):
            vals = pd.Series(pd.to_datetime(s, utc=any(getattr(v, "tzinfo", None) for v in s)))
        else:
            continue
        if isinstance(vals.dtype, pd.DatetimeTZDtype):
            vals = vals.dt.tz_convert(None)
        if (vals != vals.dt.normalize()).any():
            return "datetime"
        kind = "date"
    return kind


def data_timezone(columns: Sequence[pd.Series]) -> _dt.tzinfo | None:
    """The time zone shared by all tz-aware date-time ``columns`` (tz-aware
    ``datetime64`` dtype or aware ``datetime`` objects), else ``None`` (no aware
    column, or several different zones)."""
    zones: dict[str | None, Any] = {}
    for s in columns:
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            zones[tz_name(s.dtype.tz)] = s.dtype.tz
        elif s.dtype == object:
            for v in s.dropna():
                tz = getattr(v, "tzinfo", None)
                if tz is not None:
                    zones[tz_name(tz)] = tz
    if len(zones) != 1:
        return None
    name, tz = next(iter(zones.items()))
    try:
        return resolve_tz(name)  # a ZoneInfo for IANA names (pytz/dateutil zones too)
    except ValueError:
        return tz


# --------------------------------------------------------------------------- breaks
_UNIT_ALIASES = {
    "sec": "sec",
    "secs": "sec",
    "second": "sec",
    "seconds": "sec",
    "min": "min",
    "mins": "min",
    "minute": "min",
    "minutes": "min",
    "hour": "hour",
    "hours": "hour",
    "day": "day",
    "days": "day",
    "week": "week",
    "weeks": "week",
    "month": "month",
    "months": "month",
    "quarter": "quarter",
    "quarters": "quarter",
    "year": "year",
    "years": "year",
}
_FIXED = {"sec": 1, "min": 60, "hour": 3600, "day": _DAY, "week": 7 * _DAY}
_APPROX = {**_FIXED, "month": 30.4375 * _DAY, "year": 365.25 * _DAY}
_WIDTH_RE = re.compile(r"^\s*(\d+)?\s*([A-Za-z]+)\s*$")


def parse_date_width(spec: str) -> tuple[str, int] | pd.offsets.BaseOffset:
    """``"2 weeks"`` -> ``("week", 2)`` (units: sec, min, hour, day, week, month,
    quarter, year, singular or plural; quarters become 3 months). Anything else is
    tried as a pandas offset alias (``"MS"``, ``"QS"``, ``"7D"``, ``"h"`` ...) and
    returned as a ``DateOffset``."""
    m = _WIDTH_RE.match(spec)
    if m and m.group(2).lower() in _UNIT_ALIASES and m.group(2) == m.group(2).lower():
        k = int(m.group(1) or 1)
        if k < 1:
            raise ValueError(f"date break width must be positive, got {spec!r}")
        unit = _UNIT_ALIASES[m.group(2)]
        if unit == "quarter":
            return ("month", 3 * k)
        return (unit, k)
    try:
        return pd.tseries.frequencies.to_offset(spec)
    except ValueError:
        raise ValueError(
            f"cannot parse date break width {spec!r}; use e.g. '1 month', '2 weeks', "
            "'6 hours' or a pandas offset alias such as 'MS'"
        ) from None


def _calendar_breaks(lo: float, hi: float, unit: str, k: int) -> np.ndarray:
    """Breaks every ``k`` ``unit`` covering ``[lo, hi]`` (seconds since epoch),
    aligned to the calendar: multiples of ``k`` units since the epoch for fixed
    units (weeks start on Monday), month/year numbers divisible by ``k``."""
    if unit in _FIXED:
        width = _FIXED[unit] * k
        offset = 4 * _DAY if unit == "week" else 0  # 1970-01-05 was a Monday
        start = math.floor((lo - offset) / width + 1e-9)
        stop = math.ceil((hi - offset) / width - 1e-9)
        stop = max(stop, start + 1)
        return np.arange(start, stop + 1, dtype=float) * width + offset
    lo_dt, hi_dt = _seconds_to_dt64(np.array([lo, hi]))
    if unit == "month":
        m_lo = int(lo_dt.astype("datetime64[M]").astype("int64"))  # months since 1970-01
        m_hi = int(hi_dt.astype("datetime64[M]").astype("int64"))
        m_hi += int(hi_dt > hi_dt.astype("datetime64[M]"))
        start = math.floor(m_lo / k) * k
        stop = max(math.ceil(m_hi / k) * k, start + k)
        months = np.arange(start, stop + 1, k).astype("datetime64[M]")
    else:  # year
        y_lo = int(lo_dt.astype("datetime64[Y]").astype("int64")) + 1970
        y_hi = int(hi_dt.astype("datetime64[Y]").astype("int64")) + 1970
        y_hi += int(hi_dt > hi_dt.astype("datetime64[Y]"))
        start = math.floor(y_lo / k) * k
        stop = max(math.ceil(y_hi / k) * k, start + k)
        months = (
            (np.arange(start, stop + 1, k) - 1970).astype("datetime64[Y]").astype("datetime64[M]")
        )
    return to_epoch(months.astype("datetime64[us]"), "s")


def _offset_breaks(lo: float, hi: float, offset: pd.offsets.BaseOffset) -> np.ndarray:
    nanos = getattr(offset, "nanos", None) if isinstance(offset, pd.offsets.Tick) else None
    if nanos:
        width = nanos / 1e9
        start, stop = math.floor(lo / width + 1e-9), math.ceil(hi / width - 1e-9)
        return np.arange(start, max(stop, start + 1) + 1, dtype=float) * width
    lo_dt, hi_dt = _seconds_to_dt64(np.array([lo, hi]))
    rng = pd.date_range(pd.Timestamp(lo_dt), pd.Timestamp(hi_dt), freq=offset)
    return to_epoch(rng, "s")


_CANDIDATES: list[tuple[str, int]] = [
    *(("sec", k) for k in (1, 2, 5, 10, 15, 30)),
    *(("min", k) for k in (1, 2, 5, 10, 15, 30)),
    *(("hour", k) for k in (1, 2, 3, 6, 12)),
    ("day", 1),
    ("day", 2),
    ("week", 1),
    ("week", 2),
    *(("month", k) for k in (1, 2, 3, 6)),
    *(("year", k) for k in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000)),
]


def date_breaks_auto(lo: float, hi: float, n: int = 5, unit: str = "D") -> np.ndarray:
    """Automatic calendar breaks covering ``[lo, hi]`` (in ``unit``\\ s since the
    epoch: ``"D"`` days or ``"s"`` seconds), about ``n`` of them inside the range.

    The step is the candidate (1-30 seconds/minutes, 1-12 hours, 1-2 days, 1-2 weeks,
    1/2/3/6 months, 1/2/5/10/... years) whose number of breaks inside the range is
    closest to ``n``, preferring the larger step on ties. Date scales (``"D"``) never
    go below one day."""
    f = _SECONDS[unit]
    lo_s, hi_s = sorted((float(lo) * f, float(hi) * f))
    if not (math.isfinite(lo_s) and math.isfinite(hi_s)):
        return np.array([], dtype=float)
    span = hi_s - lo_s
    best: tuple[float, np.ndarray] | None = None
    for cand_unit, k in _CANDIDATES:
        if unit == "D" and cand_unit in ("sec", "min", "hour"):
            continue
        step = _APPROX[cand_unit] * k
        if span / step > 4 * n + 4:  # far too many; skip without generating
            continue
        b = _calendar_breaks(lo_s, hi_s, cand_unit, k)
        inside = int(np.sum((b >= lo_s - 1e-6) & (b <= hi_s + 1e-6)))
        score = abs(inside - n)
        if best is None or score <= best[0]:
            best = (score, b)
    if best is None:  # span far beyond 1000-year steps
        b = _calendar_breaks(lo_s, hi_s, "year", 10 ** max(3, int(math.log10(span / 3.15e7 / n))))
    else:
        b = best[1]
    return b / f


def _in_zone(
    brk: Callable[[float, float], np.ndarray], lo: float, hi: float, tz: _dt.tzinfo | None
) -> np.ndarray:
    """``brk(lo, hi)`` (UTC seconds) computed on ``tz``'s wall clock, mapped back to
    UTC (duplicates from daylight-saving gaps removed)."""
    if tz is None:
        return brk(lo, hi)
    wl, wh = utc_to_local([lo, hi], tz)
    return np.unique(local_to_utc(brk(wl, wh), tz))


def date_breaks_width(
    width: str, unit: str = "D", tz: Any = None
) -> Callable[[Sequence[float]], np.ndarray]:
    """Break function (for ``breaks=``/``minor_breaks=`` of a date scale) placing a
    break every ``width`` (see :func:`parse_date_width`); limits and breaks are in
    ``unit``\\ s since the epoch. ``tz`` (date-times only) aligns the breaks to
    that zone's calendar."""
    parsed = parse_date_width(width)
    f = _SECONDS[unit]
    zone = resolve_tz(tz) if unit == "s" else None

    def raw(lo: float, hi: float) -> np.ndarray:
        if isinstance(parsed, tuple):
            return _calendar_breaks(lo, hi, *parsed)
        return _offset_breaks(lo, hi, parsed)

    def brk(limits: Sequence[float]) -> np.ndarray:
        lo, hi = sorted(float(v) * f for v in limits)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return np.array([], dtype=float)
        return _in_zone(raw, lo, hi, zone) / f

    return brk


# --------------------------------------------------------------------------- labels
def _label_date(fmt: str, unit: str, tz: Any = None) -> Callable[[Any], list[str]]:
    # imported lazily: ggmanim.labels imports ggmanim.scales (circular at load time)
    from ..labels import label_date

    return label_date(fmt, unit=unit, tz=tz)


def date_format_auto(x: Any, unit: str = "D") -> str:
    """strftime format for breaks ``x`` (``unit``\\ s since the epoch), from their
    resolution: ``"%Y"`` (all on January 1st), ``"%b"`` / ``"%b %Y"`` (month starts
    within one year / across years), ``"%b %d"`` / ``"%Y-%m-%d"`` (days), ``"%H:%M"``
    (with ``"%b %d "`` in front when spanning days) or ``"%H:%M:%S"``."""
    v = np.asarray(x, dtype=float).ravel() * _SECONDS[unit]
    v = v[np.isfinite(v)]
    if not len(v):
        return "%Y-%m-%d"
    t = _seconds_to_dt64(v)
    days = t.astype("datetime64[D]")
    years = t.astype("datetime64[Y]")
    one_year = len(np.unique(years)) == 1
    one_day = len(np.unique(days)) == 1
    if np.all(t == days):
        months = t.astype("datetime64[M]")
        if np.all(t == months):
            if np.all(t == years) and len(v) > 1:
                return "%Y"
            return "%b" if one_year and len(v) > 1 else "%b %Y"
        return "%b %d" if one_year else "%Y-%m-%d"
    prefix = "" if one_day else ("%b %d " if one_year else "%Y-%m-%d ")
    if np.all(t == t.astype("datetime64[m]")):
        return prefix + "%H:%M"
    return prefix + "%H:%M:%S"


class DateTransform(Transform):
    """Identity transform on days (``unit="D"``) or seconds (``unit="s"``) since the
    epoch, with calendar breaks (:func:`date_breaks_auto`) and date labels
    (:func:`date_format_auto`). The date -> number conversion happens in
    :meth:`ScaleDatePosition.transform`. ``tz`` (date-times only) places the breaks
    on that zone's calendar and labels them in local time."""

    def __init__(self, unit: str = "D", tz: Any = None) -> None:
        if unit not in _SECONDS:
            raise ValueError(f"unit must be 'D' or 's', got {unit!r}")
        self.unit = unit
        self.name = "date" if unit == "D" else "time"
        self.tz = resolve_tz(tz) if unit == "s" else None

    def breaks(self, lo: float, hi: float, n: int = 5) -> np.ndarray:
        def auto(a: float, b: float) -> np.ndarray:
            return date_breaks_auto(a, b, n, unit=self.unit)

        return _in_zone(auto, lo, hi, self.tz)

    def format(self, x: np.ndarray) -> list[str]:
        wall = x if self.tz is None else utc_to_local(x, self.tz)
        return _label_date(date_format_auto(wall, self.unit), self.unit, self.tz)(x)


# --------------------------------------------------------------------------- scale
def _to_num(v: Any, unit: str) -> float | None:
    if v is None:
        return None
    return float(to_epoch(v, unit)[0])


class ScaleDatePosition(ScaleContinuousPosition):
    """Date (``unit="D"``, days since 1970-01-01) or date-time (``unit="s"``,
    seconds) x/y position scale. See the module docstring.

    Extra parameters over :class:`ScaleContinuousPosition`: ``date_breaks`` and
    ``date_minor_breaks`` (width strings such as ``"1 month"``), ``date_labels`` (a
    strftime format). ``limits``, ``breaks`` and ``minor_breaks`` sequences may hold
    dates, ``datetime64`` values or ISO strings. ``trans`` is fixed. ``timezone``
    (date-times only): see the module docstring.
    """

    is_date = True

    def __init__(
        self,
        aesthetics,
        name: str | None = None,
        breaks: Sequence[Any] | Callable | None = None,
        labels: Sequence[str] | Callable | None = None,
        limits: Sequence[Any] | None = None,
        expand=None,
        oob: str | Callable = "censor",
        n_breaks: int = 5,
        minor_breaks: Sequence[Any] | Callable | bool | None = None,
        position: str | None = None,
        *,
        unit: str = "D",
        date_breaks: str | None = None,
        date_labels: str | None = None,
        date_minor_breaks: str | None = None,
        timezone: Any = None,
    ) -> None:
        if timezone is not None and unit != "s":
            raise ValueError("timezone= applies to date-time scales (scale_*_datetime) only")
        tz = resolve_tz(timezone)
        trans = DateTransform(unit, tz)
        self.unit = unit
        self.timezone = tz
        self._date_specs = (date_breaks, date_labels, date_minor_breaks)
        if date_breaks is not None:
            breaks = date_breaks_width(date_breaks, unit, tz)
        elif breaks is not None and not callable(breaks):
            breaks = tuple(to_epoch(list(breaks), unit).tolist())
        if date_labels is not None:
            labels = _label_date(date_labels, unit, tz)
        if date_minor_breaks is not None:
            minor_breaks = date_breaks_width(date_minor_breaks, unit, tz)
        elif minor_breaks is not None and not callable(minor_breaks) and minor_breaks is not False:
            if minor_breaks is not True:
                minor_breaks = tuple(to_epoch(list(minor_breaks), unit).tolist())
        if limits is not None:
            if isinstance(limits, (str, _dt.date)) or len(limits) != 2:
                raise ValueError(f"limits must be (lo, hi), got {limits!r}")
            limits = (_to_num(limits[0], unit), _to_num(limits[1], unit))
        super().__init__(
            aesthetics,
            name=name,
            breaks=breaks,
            labels=labels,
            limits=limits,
            expand=expand,
            oob=oob,
            trans=trans,
            n_breaks=n_breaks,
            minor_breaks=minor_breaks,
            position=position,
        )

    def transform(self, values: Any) -> np.ndarray:
        """Dates / date-times / ISO strings / numbers -> days or seconds since epoch."""
        return to_epoch(values, self.unit)

    def set_timezone(self, tz: Any) -> None:
        """Use time zone ``tz`` for breaks and labels (date-time scales only; call on a
        :meth:`clone`, as the build does with the zone of tz-aware data). Rebuilds the
        ``date_breaks``/``date_labels``/``date_minor_breaks`` helpers; user
        ``breaks``/``labels`` functions are left alone."""
        if self.unit != "s":
            return
        tz = resolve_tz(tz)
        self.timezone = tz
        self.trans = DateTransform(self.unit, tz)
        date_breaks, date_labels, date_minor_breaks = self._date_specs
        if date_breaks is not None:
            self.breaks = date_breaks_width(date_breaks, self.unit, tz)
        if date_labels is not None:
            self.labels = _label_date(date_labels, self.unit, tz)
        if date_minor_breaks is not None:
            self.minor_breaks = date_breaks_width(date_minor_breaks, self.unit, tz)


# --------------------------------------------------------------------------- constructors
def _date_scale(axis: str, unit: str, **kw) -> ScaleDatePosition:
    aes = X_AESTHETICS if axis == "x" else Y_AESTHETICS
    return ScaleDatePosition(aes, unit=unit, **kw)


@register
def scale_x_date(
    name=None,
    breaks=None,
    date_breaks=None,
    labels=None,
    date_labels=None,
    minor_breaks=None,
    date_minor_breaks=None,
    limits=None,
    expand=None,
    oob="censor",
    n_breaks=5,
    position=None,
):
    """x scale for dates (days since 1970-01-01). ``date_breaks="1 month"``,
    ``date_labels="%b %Y"``, ``date_minor_breaks="1 week"``; ``limits`` may be dates or
    ISO strings. See :class:`ScaleDatePosition`."""
    return _date_scale(
        "x", "D", name=name, breaks=breaks, date_breaks=date_breaks, labels=labels,
        date_labels=date_labels, minor_breaks=minor_breaks,
        date_minor_breaks=date_minor_breaks, limits=limits, expand=expand, oob=oob,
        n_breaks=n_breaks, position=position,
    )  # fmt: skip


@register
def scale_y_date(
    name=None,
    breaks=None,
    date_breaks=None,
    labels=None,
    date_labels=None,
    minor_breaks=None,
    date_minor_breaks=None,
    limits=None,
    expand=None,
    oob="censor",
    n_breaks=5,
    position=None,
):
    """y scale for dates. See :func:`scale_x_date`."""
    return _date_scale(
        "y", "D", name=name, breaks=breaks, date_breaks=date_breaks, labels=labels,
        date_labels=date_labels, minor_breaks=minor_breaks,
        date_minor_breaks=date_minor_breaks, limits=limits, expand=expand, oob=oob,
        n_breaks=n_breaks, position=position,
    )  # fmt: skip


@register
def scale_x_datetime(
    name=None,
    breaks=None,
    date_breaks=None,
    labels=None,
    date_labels=None,
    minor_breaks=None,
    date_minor_breaks=None,
    limits=None,
    expand=None,
    oob="censor",
    n_breaks=5,
    position=None,
    timezone=None,
):
    """x scale for date-times (seconds since 1970-01-01 UTC). ``date_breaks="6 hours"``,
    ``date_labels="%H:%M"``; ``timezone="Europe/Paris"`` puts breaks and labels in that
    zone (default: the zone of tz-aware data, else UTC); see :func:`scale_x_date`."""
    return _date_scale(
        "x", "s", name=name, breaks=breaks, date_breaks=date_breaks, labels=labels,
        date_labels=date_labels, minor_breaks=minor_breaks,
        date_minor_breaks=date_minor_breaks, limits=limits, expand=expand, oob=oob,
        n_breaks=n_breaks, position=position, timezone=timezone,
    )  # fmt: skip


@register
def scale_y_datetime(
    name=None,
    breaks=None,
    date_breaks=None,
    labels=None,
    date_labels=None,
    minor_breaks=None,
    date_minor_breaks=None,
    limits=None,
    expand=None,
    oob="censor",
    n_breaks=5,
    position=None,
    timezone=None,
):
    """y scale for date-times. See :func:`scale_x_datetime`."""
    return _date_scale(
        "y", "s", name=name, breaks=breaks, date_breaks=date_breaks, labels=labels,
        date_labels=date_labels, minor_breaks=minor_breaks,
        date_minor_breaks=date_minor_breaks, limits=limits, expand=expand, oob=oob,
        n_breaks=n_breaks, position=position, timezone=timezone,
    )  # fmt: skip
