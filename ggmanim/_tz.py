"""Time-zone helpers for date-time scales and labels (stdlib :mod:`zoneinfo` only).

Date-time scales keep their transformed space in **UTC** seconds since the epoch;
a ``timezone`` only changes where breaks fall and how they are labelled. These
helpers move between UTC seconds and *wall-clock* seconds (the local time read as
if it were UTC), so calendar arithmetic can be done on the wall clock.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

__all__ = ["resolve_tz", "tz_name", "utc_to_local", "local_to_utc"]

_EPOCH_UTC = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
_EPOCH = _dt.datetime(1970, 1, 1)


def resolve_tz(tz: Any) -> _dt.tzinfo | None:
    """``None`` -> ``None``; an IANA name (``"Europe/London"``, ``"UTC"``) -> a
    :class:`~zoneinfo.ZoneInfo`; a :class:`datetime.tzinfo` passes through."""
    if tz is None or isinstance(tz, _dt.tzinfo):
        return tz
    if isinstance(tz, str):
        try:
            return ZoneInfo(tz)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(f"unknown time zone {tz!r}; use an IANA name such as 'UTC'") from None
    raise TypeError(f"timezone must be a string or tzinfo, got {type(tz).__name__}")


def tz_name(tz: _dt.tzinfo | None) -> str | None:
    """A comparable name for ``tz`` (its IANA key when it has one)."""
    if tz is None:
        return None
    return str(getattr(tz, "key", None) or tz)


def _map(sec: Any, f) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(sec, dtype=float)).ravel()
    return np.array([f(s) if math.isfinite(s) else s for s in arr], dtype=float)


def utc_to_local(sec: Any, tz: _dt.tzinfo) -> np.ndarray:
    """UTC seconds since the epoch -> wall-clock seconds in ``tz``."""

    def one(s: float) -> float:
        off = (_EPOCH_UTC + _dt.timedelta(seconds=s)).astimezone(tz).utcoffset()
        return s + (off.total_seconds() if off else 0.0)

    return _map(sec, one)


def local_to_utc(sec: Any, tz: _dt.tzinfo) -> np.ndarray:
    """Wall-clock seconds in ``tz`` -> UTC seconds. Ambiguous times (clocks going
    back) take the first occurrence; non-existent ones (clocks going forward) use the
    offset before the change."""

    def one(s: float) -> float:
        off = (_EPOCH + _dt.timedelta(seconds=s)).replace(tzinfo=tz).utcoffset()
        return s - (off.total_seconds() if off else 0.0)

    return _map(sec, one)
