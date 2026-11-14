"""Secondary position axes: ``sec_axis`` / ``dup_axis`` (ggplot2's ``AxisSecondary``).

A secondary axis is attached to a continuous position scale::

    scale_y_continuous(name="°C", sec_axis=sec_axis(lambda c: c * 1.8 + 32, name="°F"))
    scale_x_log10(sec_axis=dup_axis())

and drawn by the coord on the side opposite the primary axis (right for a
left y axis, top for a bottom x axis; swapped by ``position="top"/"right"``).

The secondary axis has no range of its own. :meth:`AxisSecondary.break_info`
follows ggplot2: sample the primary range (1000 points), apply ``transform`` to
get the secondary values, compute breaks for the secondary range with the
primary scale's transform (so a log primary gets log breaks on a
:func:`dup_axis`), then map each break back onto the primary axis by monotone
interpolation. ``transform`` must therefore be monotone over the primary range.

Date and date-time scales (``scale_x_date(sec_axis=...)``) bind the axis with
:meth:`AxisSecondary.bind`: ``transform`` receives the primary values as numbers
(days or seconds since 1970-01-01, so ``lambda t: t + 3600`` shifts a date-time
axis by an hour, as ``. + 3600`` does in ggplot2) and may return numbers or
dates; ``breaks`` may be dates or ISO strings; date-aware ``labels`` formatters
get the scale's unit and time zone. Breaks and labels then come from the date
scale's transform (calendar breaks, date labels).

Pure numpy; no manim imports.
"""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .continuous import IdentityTransform, ScaleContinuous

__all__ = ["AxisSecondary", "Derive", "derive", "sec_axis", "dup_axis"]
__toplevel__ = ["sec_axis", "dup_axis", "derive"]

#: Number of points sampled across the primary range (as ggplot2's ``detail``).
DETAIL = 1000


class Derive:
    """Sentinel (ggplot2's ``derive()``): inherit the setting from the primary scale."""

    _instance: Derive | None = None

    def __new__(cls) -> Derive:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "derive()"


_DERIVE = Derive()


def derive() -> Derive:
    """Inherit ``name``/``breaks``/``labels`` of a secondary axis from the primary scale."""
    return Derive()


def _is_derive(v: Any) -> bool:
    return isinstance(v, Derive)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


class _RawBreaks(tuple):
    """Secondary breaks given as dates / strings, waiting for :meth:`AxisSecondary.bind`."""


class AxisSecondary:
    """A secondary axis specification (see the module docstring).

    Parameters
    ----------
    transform : callable ``f(primary data values) -> secondary values`` (vectorised
        over numpy arrays; scalar-only functions are applied element-wise), or
        ``None`` for the identity. Must be monotone over the primary range.
    name : axis title; ``None`` = no title, :func:`derive` = the primary title.
    breaks : ``None`` (automatic), a sequence of *secondary* data values (dates or ISO
        strings on a date scale), a callable
        ``f((lo, hi)) -> breaks`` on the secondary range, or :func:`derive` (the
        primary scale's ``breaks`` setting applied to the secondary range).
    labels : ``None`` (the primary scale's label formatter if it is a callable,
        else the transform's), a sequence of strings (one per user break), a
        callable ``f(breaks) -> list[str]``, or :func:`derive` (the primary's).
    """

    def __init__(
        self,
        transform: Callable | None = None,
        name: Any = None,
        breaks: Any = None,
        labels: Any = None,
    ) -> None:
        if transform is not None and not callable(transform):
            raise TypeError(f"sec_axis transform must be callable, got {transform!r}")
        self.transform = transform
        self.name = name
        #: converts dates (breaks, transform results) to numbers; set by :meth:`bind`
        self._convert: Callable[[Any], Any] | None = None
        if breaks is not None and not callable(breaks) and not _is_derive(breaks):
            raw = list(np.asarray(breaks, dtype=object).ravel())
            if all(_is_number(b) for b in raw):
                breaks = tuple(float(b) for b in raw)
            else:  # dates / ISO strings: converted when bound to a date scale
                breaks = _RawBreaks(raw)
        self.breaks = breaks
        if labels is not None and not callable(labels) and not _is_derive(labels):
            labels = tuple(str(lab) for lab in labels)
            if isinstance(breaks, (tuple, _RawBreaks)) and len(labels) != len(breaks):
                raise ValueError("sec_axis labels must have the same length as breaks")
        self.labels = labels

    def __repr__(self) -> str:
        return (
            f"sec_axis(transform={self.transform!r}, name={self.name!r}, "
            f"breaks={self.breaks!r}, labels={self.labels!r})"
        )

    def bind(
        self,
        convert: Callable[[Any], Any] | None = None,
        bind_labels: Callable[[Any], Any] | None = None,
    ) -> AxisSecondary:
        """A copy of this axis for a scale whose values need converting (date and
        date-time scales). ``convert(values) -> float array`` turns dates / ISO
        strings into the scale's numbers: it is applied to ``breaks`` and to
        ``transform`` results that are not numeric. ``bind_labels(labels)``
        completes a callable ``labels`` formatter (e.g. a date-aware one with the
        scale's unit and time zone)."""
        new = copy.copy(self)
        new._convert = convert
        if isinstance(self.breaks, _RawBreaks) and convert is not None:
            new.breaks = tuple(float(v) for v in np.asarray(convert(list(self.breaks)), float))
        if bind_labels is not None and callable(self.labels) and not _is_derive(self.labels):
            new.labels = bind_labels(self.labels)
        return new

    # ------------------------------------------------------------------ helpers
    def _to_float(self, res: Any) -> np.ndarray:
        arr = np.asarray(res)
        if arr.dtype.kind in "biuf" or self._convert is None:
            return np.asarray(res, dtype=float)
        return np.asarray(self._convert(res), dtype=float)

    def _apply(self, x: np.ndarray) -> np.ndarray:
        if self.transform is None:
            return np.asarray(x, dtype=float)
        try:
            out = self._to_float(self.transform(x))
            if out.shape != x.shape:
                out = np.broadcast_to(out, x.shape).astype(float)
        except (TypeError, ValueError):
            out = np.array(
                [float(np.ravel(self._to_float(self.transform(float(v))))[0]) for v in x],
                dtype=float,
            )
        return out

    def title(self, primary_title: Any) -> Any:
        """The secondary axis title given the primary one (resolved by the coord)."""
        return primary_title if _is_derive(self.name) else self.name

    def _temp_scale(self, scale: Any, trans: Any) -> ScaleContinuous:
        breaks = getattr(scale, "breaks", None) if _is_derive(self.breaks) else self.breaks
        if _is_derive(self.labels):
            labels = getattr(scale, "labels", None)
        elif self.labels is None:
            plab = getattr(scale, "labels", None)
            labels = plab if callable(plab) else None
        else:
            labels = self.labels
        if not (labels is None or callable(labels)) and not isinstance(breaks, tuple):
            labels = None  # fixed label strings need fixed breaks to pair with
        return ScaleContinuous(
            "sec",
            breaks=breaks,
            labels=labels,
            trans=trans,
            n_breaks=getattr(scale, "n_breaks", 5),
        )

    # ------------------------------------------------------------------ breaks
    def break_info(
        self, scale: Any, primary_range: Sequence[float]
    ) -> tuple[tuple[float, ...], tuple[str, ...]]:
        """``(positions, labels)`` of the secondary axis.

        ``primary_range`` is the drawn primary range in the scale's *transformed*
        space (as in ``PanelParams``); the positions returned are in that same
        space (where the ticks go on the primary axis), the labels are the
        secondary data-space strings. A non-monotone transform warns and gives no
        breaks.
        """
        if getattr(scale, "is_discrete", False):
            raise TypeError("sec_axis is only supported on continuous position scales")
        if isinstance(self.breaks, _RawBreaks):
            raise TypeError(
                "sec_axis breaks that are not numbers (dates, strings) need a date or "
                f"date-time scale, got {list(self.breaks)[:3]!r}"
            )
        lo, hi = float(min(primary_range)), float(max(primary_range))
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            return (), ()
        ptrans = getattr(scale, "trans", None) or IdentityTransform()
        tgrid = np.linspace(lo, hi, DETAIL)
        with np.errstate(all="ignore"):
            data = np.asarray(ptrans.inverse(tgrid), dtype=float)
            sec = self._apply(data)
        if not np.all(np.isfinite(sec)):
            warnings.warn(
                "sec_axis transform gave non-finite values over the primary range; "
                "no secondary axis drawn",
                stacklevel=2,
            )
            return (), ()
        d = np.diff(sec)
        if not (np.all(d > 0) or np.all(d < 0)):
            warnings.warn(
                "sec_axis transform must be strictly monotone over the primary range; "
                "no secondary axis drawn",
                stacklevel=2,
            )
            return (), ()

        # breaks on the secondary range with the primary's transform (log -> log
        # breaks), falling back to identity when the secondary range leaves its domain
        trans = ptrans
        with np.errstate(all="ignore"):
            tsec = np.asarray(trans.transform(sec), dtype=float)
        dt = np.diff(tsec)
        if not (np.all(np.isfinite(tsec)) and (np.all(dt > 0) or np.all(dt < 0))):
            trans = IdentityTransform()
            tsec, dt = sec, d
        tmp = self._temp_scale(scale, trans)
        limits = (float(tsec.min()), float(tsec.max()))
        tbreaks = np.asarray(tmp.get_breaks(limits), dtype=float)
        labels = tmp.get_labels(limits)
        if not len(tbreaks):
            return (), ()

        # back onto the primary axis: interpolate (transformed secondary) -> primary
        xp, fp = (tsec, tgrid) if dt[0] > 0 else (tsec[::-1], tgrid[::-1])
        pos = np.interp(tbreaks, xp, fp)
        order = np.argsort(pos, kind="stable")
        return (
            tuple(float(pos[i]) for i in order),
            tuple(str(labels[i]) for i in order),
        )


def sec_axis(
    transform: Callable | None = None,
    name: Any = None,
    breaks: Any = None,
    labels: Any = None,
    *,
    trans: Callable | None = None,
) -> AxisSecondary:
    """A secondary axis whose values are ``transform(primary values)``.

    ``scale_y_continuous(sec_axis=sec_axis(lambda c: c * 1.8 + 32, name="°F"))``
    puts Fahrenheit ticks on the right of a Celsius axis. ``trans=`` is accepted as
    an alias of ``transform`` (ggplot2 < 3.5). See :class:`AxisSecondary`.
    """
    if trans is not None:
        if transform is not None:
            raise TypeError("give sec_axis either transform= or trans=, not both")
        transform = trans
    return AxisSecondary(transform, name=name, breaks=breaks, labels=labels)


def dup_axis(
    transform: Callable | None = None,
    name: Any = _DERIVE,
    breaks: Any = _DERIVE,
    labels: Any = _DERIVE,
    *,
    trans: Callable | None = None,
) -> AxisSecondary:
    """A secondary axis that mirrors the primary one (title, breaks and labels
    :func:`derive` from the primary scale unless given)."""
    return sec_axis(transform, name=name, breaks=breaks, labels=labels, trans=trans)
