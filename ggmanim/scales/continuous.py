"""Continuous scales and scale transforms.

Pure numpy; no manim imports.

Spaces
------
* **data space** -- raw column values (e.g. ``3 .. 2500``).
* **transformed space** -- after the scale's :class:`Transform` (e.g. ``log10`` gives
  ``0.48 .. 3.40``). Position scales train, compute limits/breaks and map in this
  space; the coord draws a *linear* manim ``Axes`` over it and uses the labels
  (data-space strings) for tick text. Manim's ``LogBase`` is never used.

Position-scale API used by the build / coord
--------------------------------------------
``train(values)``            data-space values -> update trained range.
``train_transformed(tv)``    same, for values already in transformed space.
``transform(values)``        data -> transformed (no oob handling).
``map(values)``              data -> transformed, out-of-bounds handled by ``oob``.
``map_transformed(tv)``      oob handling only, for already-transformed values.
``get_limits()``             ``(lo, hi)`` transformed, ascending, *without* expansion.
``get_expanded_limits()``    ``(lo, hi)`` incl. ``expand`` -- the axis range to draw.
``dimension()``              alias of ``get_expanded_limits()`` (ggplot2 name).
``get_breaks(limits=None)``  ndarray of major breaks in transformed space, censored
                             to ``limits`` (default ``get_limits()``; the coord may pass
                             the expanded limits to get breaks in the padding too).
``get_labels(limits=None)``  list[str], one per ``get_breaks(limits)``, data space.
``get_minor_breaks(limits=None)``  ndarray of minor breaks (transformed space, majors
                             excluded) for the minor grid.
``position``                 (position scales) axis side: ``"bottom"``/``"top"`` for x,
                             ``"left"``/``"right"`` for y.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..registry import register
from .base import Scale
from .breaks import (
    expand_range,
    format_numbers,
    log_breaks,
    log_minor_breaks,
    pretty,
    regular_minor_breaks,
)

__all__ = [
    "Transform",
    "IdentityTransform",
    "LogTransform",
    "SqrtTransform",
    "ReverseTransform",
    "get_transform",
    "censor",
    "squish",
    "ScaleContinuous",
    "ScaleContinuousPosition",
    "scale_x_continuous",
    "scale_y_continuous",
    "scale_x_log10",
    "scale_y_log10",
    "scale_x_sqrt",
    "scale_y_sqrt",
    "scale_x_reverse",
    "scale_y_reverse",
    "xlim",
    "ylim",
]

X_AESTHETICS = ("x", "xmin", "xmax", "xend", "xintercept", "xlower", "xmiddle", "xupper", "x0")
Y_AESTHETICS = ("y", "ymin", "ymax", "yend", "yintercept", "ylower", "ymiddle", "yupper", "y0")


def _as_float_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind not in "biuf":
        try:
            arr = arr.astype(float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"continuous scale got non-numeric values (dtype {arr.dtype}); "
                "use a discrete scale for categorical data"
            ) from exc
    return np.atleast_1d(arr.astype(float, copy=False))


# --------------------------------------------------------------------------- transforms
class Transform:
    """A scale transformation (``scales::trans``).

    ``transform``/``inverse`` map data <-> transformed space (vectorised; values
    outside ``domain`` become NaN). ``breaks(lo, hi, n)`` returns break candidates in
    *data* space for the data-space range; ``format(x)`` turns data-space breaks
    into label strings.
    """

    name = "identity"
    domain: tuple[float, float] = (-math.inf, math.inf)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)

    def breaks(self, lo: float, hi: float, n: int = 5) -> np.ndarray:
        return pretty(lo, hi, n)

    def format(self, x: np.ndarray) -> list[str]:
        return format_numbers(x)

    def _in_domain(self, x: np.ndarray) -> np.ndarray:
        lo, hi = self.domain
        return np.where((x >= lo) & (x <= hi), x, np.nan)

    def __repr__(self) -> str:
        return f"<Transform {self.name}>"


class IdentityTransform(Transform):
    name = "identity"


class LogTransform(Transform):
    """Logarithm with the given base. Non-positive values map to NaN."""

    def __init__(self, base: float = 10) -> None:
        self.base = float(base)
        self.name = "log10" if base == 10 else ("log" if base == math.e else f"log-{base:g}")
        self.domain = (math.ulp(0.0), math.inf)  # smallest positive float

    def transform(self, x):
        x = self._in_domain(np.asarray(x, dtype=float))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log(x) / math.log(self.base)

    def inverse(self, x):
        return self.base ** np.asarray(x, dtype=float)

    def breaks(self, lo, hi, n=5):
        return log_breaks(lo, hi, n, base=self.base)


class SqrtTransform(Transform):
    """Square root. Negative values map to NaN."""

    name = "sqrt"
    domain = (0.0, math.inf)

    def transform(self, x):
        x = self._in_domain(np.asarray(x, dtype=float))
        with np.errstate(invalid="ignore"):
            return np.sqrt(x)

    def inverse(self, x):
        return np.asarray(x, dtype=float) ** 2

    def breaks(self, lo, hi, n=5):
        return pretty(max(lo, 0.0), hi, n)


class ReverseTransform(Transform):
    """Negation, so larger data values sit lower/left on the axis."""

    name = "reverse"

    def transform(self, x):
        return 0.0 - np.asarray(x, dtype=float)  # 0.0 - x avoids -0.0

    def inverse(self, x):
        return 0.0 - np.asarray(x, dtype=float)


_TRANSFORMS: dict[str, Callable[[], Transform]] = {
    "identity": IdentityTransform,
    "log10": lambda: LogTransform(10),
    "log2": lambda: LogTransform(2),
    "log": lambda: LogTransform(math.e),
    "sqrt": SqrtTransform,
    "reverse": ReverseTransform,
}


def get_transform(trans: str | Transform | None) -> Transform:
    """Look up a transform by name (identity, log10, log2, log, sqrt, reverse) or
    pass a :class:`Transform` instance through."""
    if trans is None:
        return IdentityTransform()
    if isinstance(trans, Transform):
        return trans
    try:
        return _TRANSFORMS[trans]()
    except KeyError:
        raise ValueError(
            f"unknown transform {trans!r}; choose from {sorted(_TRANSFORMS)}"
        ) from None


# --------------------------------------------------------------------------- oob
def censor(x: np.ndarray, range: tuple[float, float]) -> np.ndarray:
    """Replace values outside ``range`` with NaN (infinite values are kept)."""
    x = np.asarray(x, dtype=float)
    lo, hi = range
    tol = 1e-10 * max(abs(hi - lo), abs(lo), abs(hi), 1e-300)
    bad = np.isfinite(x) & ((x < lo - tol) | (x > hi + tol))
    return np.where(bad, np.nan, x)


def squish(x: np.ndarray, range: tuple[float, float]) -> np.ndarray:
    """Clip finite values into ``range``."""
    x = np.asarray(x, dtype=float)
    return np.where(np.isfinite(x), np.clip(x, range[0], range[1]), x)


def _keep(x: np.ndarray, range: tuple[float, float]) -> np.ndarray:
    return np.asarray(x, dtype=float)


_OOB = {"censor": censor, "squish": squish, "keep": _keep}


# --------------------------------------------------------------------------- scale
_UNSET = object()


class ScaleContinuous(Scale):
    """Continuous scale with a transform, user limits/breaks/labels, expansion
    and an out-of-bounds policy.

    Parameters
    ----------
    aesthetics : str or sequence of str
    name : axis/legend title (``None`` = default from labs / mapping).
    breaks : ``None`` (automatic), a sequence of data-space values (``[]`` for no
        breaks) or a callable ``f((lo, hi)) -> data-space breaks`` receiving the
        data-space limits.
    labels : ``None`` (transform's formatter), a sequence of strings (same length as
        a user ``breaks`` sequence) or a callable ``f(data_breaks) -> list[str]``.
    limits : ``None`` or ``(lo, hi)`` in data space; either end may be ``None`` to use
        the trained value. Values outside are handled by ``oob``.
    expand : ``None`` (default 5% each side), ``False``, ``(mult, add)`` or
        ``(mult_lo, add_lo, mult_hi, add_hi)``; see :func:`.breaks.expansion`.
        Applied in transformed space.
    oob : ``"censor"`` (default; out-of-range -> NaN), ``"squish"``, ``"keep"`` or a
        callable ``f(transformed_values, (lo, hi)) -> values``.
    trans : transform name or :class:`Transform`.
    n_breaks : desired number of automatic breaks.
    minor_breaks : ``None`` (default: one minor break between adjacent majors; for log
        transforms the ``2 .. base-1`` multiples of each power), ``False`` or ``[]``
        (none), a sequence of data-space values, or a callable
        ``f((lo, hi)) -> data-space values`` (e.g. :func:`ggmanim.labels.breaks_width`).
        Used for the minor grid (see :meth:`get_minor_breaks`).
    """

    def __init__(
        self,
        aesthetics,
        name: str | None = None,
        breaks: Sequence[float] | Callable | None = None,
        labels: Sequence[str] | Callable | None = None,
        limits: Sequence[float | None] | None = None,
        expand=None,
        oob: str | Callable = "censor",
        trans: str | Transform | None = "identity",
        n_breaks: int = 5,
        minor_breaks: Sequence[float] | Callable | bool | None = None,
    ) -> None:
        super().__init__(aesthetics, name)
        self.trans = get_transform(trans)
        if limits is not None:
            limits = tuple(limits)
            if len(limits) != 2:
                raise ValueError(
                    f"limits must be (lo, hi), got {limits!r}; use breaks= to set tick positions"
                )
        self.limits = limits
        if breaks is not None and not callable(breaks):
            breaks = tuple(float(b) for b in np.asarray(breaks, dtype=float).ravel())
        self.breaks = breaks
        if labels is not None and not callable(labels):
            labels = tuple(str(lab) for lab in labels)
            if isinstance(self.breaks, tuple) and len(labels) != len(self.breaks):
                raise ValueError("labels must have the same length as breaks")
        self.labels = labels
        self.expand = expand
        if isinstance(oob, str):
            if oob not in _OOB:
                raise ValueError(f"unknown oob {oob!r}; choose from {sorted(_OOB)}")
            oob = _OOB[oob]
        self.oob = oob
        self.n_breaks = int(n_breaks)
        if minor_breaks is False:
            minor_breaks = ()
        elif minor_breaks is True:
            minor_breaks = None
        elif minor_breaks is not None and not callable(minor_breaks):
            minor_breaks = tuple(float(b) for b in np.asarray(minor_breaks, dtype=float).ravel())
        self.minor_breaks = minor_breaks
        self._range: tuple[float, float] | None = None

    # ------------------------------------------------------------- training
    def reset(self) -> None:
        self._range = None

    def transform(self, values: Any) -> np.ndarray:
        """Data space -> transformed space (no oob handling)."""
        return self.trans.transform(_as_float_array(values))

    def train(self, values: Any) -> None:
        self.train_transformed(self.transform(values))

    def train_transformed(self, values: Any) -> None:
        x = _as_float_array(values)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return
        lo, hi = float(x.min()), float(x.max())
        if self._range is not None:
            lo, hi = min(lo, self._range[0]), max(hi, self._range[1])
        self._range = (lo, hi)

    @property
    def range(self) -> tuple[float, float] | None:
        """Trained range in transformed space (``None`` if untrained)."""
        return self._range

    # ------------------------------------------------------------- mapping
    def map(self, values: Any) -> np.ndarray:
        """Data space -> transformed space, then apply ``oob`` against the limits."""
        return self.map_transformed(self.transform(values))

    def map_transformed(self, values: Any) -> np.ndarray:
        x = _as_float_array(values)
        limits = self.get_limits()
        if limits is None:
            return x
        return np.asarray(self.oob(x, limits), dtype=float)

    # ------------------------------------------------------------- limits
    def get_limits(self) -> tuple[float, float] | None:
        """``(lo, hi)`` in transformed space, ascending, without expansion."""
        if self.limits is None:
            return self._range
        user = [
            float(self.trans.transform(np.array([v], dtype=float))[0]) if v is not None else None
            for v in self.limits
        ]
        if any(v is not None and not math.isfinite(v) for v in user):
            raise ValueError(f"limits {self.limits!r} are outside the domain of {self.trans.name}")
        if None in user:
            if self._range is None:
                known = [v for v in user if v is not None]
                if not known:
                    return None
                return (known[0], known[0])
            # Fill a missing end from the trained range. limits[0] is the smallest
            # data value, which a decreasing transform (reverse) maps to the top.
            tr_lo, tr_hi = self._range
            inc = self._is_increasing()
            user = [
                v if v is not None else (tr_lo if (i == 0) == inc else tr_hi)
                for i, v in enumerate(user)
            ]
        lo, hi = sorted(user)  # type: ignore[type-var]
        return (float(lo), float(hi))  # type: ignore[arg-type]  # Nones filled above

    def _is_increasing(self) -> bool:
        a, b = self.trans.transform(np.array([1.0, 2.0]))
        return bool(b > a)

    def get_expanded_limits(self, expand=_UNSET) -> tuple[float, float] | None:
        """Limits plus expansion, in transformed space (the axis range to draw)."""
        limits = self.get_limits()
        if limits is None:
            return None
        return expand_range(limits, self.expand if expand is _UNSET else expand)

    def dimension(self, expand=_UNSET) -> tuple[float, float] | None:
        """ggplot2-style alias for :meth:`get_expanded_limits`."""
        return self.get_expanded_limits(expand)

    def _data_limits(self, limits: tuple[float, float]) -> tuple[float, float]:
        d = self.trans.inverse(np.asarray(limits, dtype=float))
        return (float(np.min(d)), float(np.max(d)))

    # ------------------------------------------------------------- breaks/labels
    def _default_breaks(self, dlo: float, dhi: float, limits) -> np.ndarray:
        """``trans.breaks`` asking for more intervals while fewer than
        ``n_breaks - 1`` breaks fall inside ``limits`` (R's ``pretty`` snaps outward,
        so e.g. 0..13 would otherwise keep only 0, 5, 10)."""
        want = max(2, self.n_breaks - 1)
        # log breaks already pick enough powers/sub-breaks (scales::log_breaks)
        tries = 1 if isinstance(self.trans, LogTransform) else 4
        best = None
        for n in range(self.n_breaks, self.n_breaks + tries):
            b = np.asarray(self.trans.breaks(dlo, dhi, n), dtype=float)
            t = self.trans.transform(b)
            inside = int(np.sum(np.isfinite(t) & np.isfinite(censor(t, limits))))
            if best is None:
                best = b
            if inside >= want:
                return b
        assert best is not None
        return best

    def _breaks_and_labels(self, limits=None) -> tuple[np.ndarray, list[str]]:
        if limits is None:
            limits = self.get_limits()
        if limits is None or self.breaks == ():
            return np.array([], dtype=float), []
        limits = (float(min(limits)), float(max(limits)))
        dlo, dhi = self._data_limits(limits)

        if self.breaks is None and limits[0] == limits[1]:
            # zero-width range: a single break at the value (as ggplot2 does)
            data_breaks = np.array([dlo])
        elif self.breaks is None:
            if self.trans.domain[0] > -math.inf:
                dlo = max(dlo, self.trans.domain[0])
            if not (math.isfinite(dlo) and math.isfinite(dhi)):
                return np.array([], dtype=float), []
            data_breaks = self._default_breaks(dlo, dhi, limits)
        elif callable(self.breaks):
            data_breaks = np.asarray(self.breaks((dlo, dhi)), dtype=float).ravel()
        else:
            data_breaks = np.asarray(self.breaks, dtype=float)

        tbreaks = self.trans.transform(data_breaks)
        keep = np.isfinite(tbreaks) & np.isfinite(censor(tbreaks, limits))
        tbreaks, kept = tbreaks[keep], data_breaks[keep]

        if self.labels is None:
            labels = self.trans.format(kept)
        elif callable(self.labels):
            labels = [str(lab) for lab in self.labels(kept)]
            if len(labels) != len(kept):
                raise ValueError("labels() must return one label per break")
        else:
            if len(self.labels) != len(data_breaks):
                raise ValueError("labels must have the same length as breaks")
            labels = [lab for lab, k in zip(self.labels, keep) if k]
        return tbreaks, labels

    def get_breaks(self, limits=None) -> np.ndarray:
        """Major breaks in transformed space, censored to ``limits`` (transformed;
        default ``get_limits()``)."""
        return self._breaks_and_labels(limits)[0]

    def get_labels(self, limits=None) -> list[str]:
        """Labels (data-space strings) for ``get_breaks(limits)``."""
        return self._breaks_and_labels(limits)[1]

    def has_default_breaks(self) -> bool:
        """Whether the major breaks are automatic (``breaks=None``), i.e. chosen
        from ``n_breaks`` rather than given by the user."""
        return self.breaks is None

    def with_n_breaks(self, n: int) -> ScaleContinuous:
        """A shallow copy of this scale whose automatic breaks target ``n`` breaks
        (the coord asks for fewer when the default ones would crowd a small panel).
        User-given ``breaks`` are kept as they are; the scale itself is not mutated."""
        out = copy.copy(self)
        out.n_breaks = max(1, int(n))
        return out

    def get_minor_breaks(self, limits=None) -> np.ndarray:
        """Minor breaks in transformed space within ``limits`` (transformed; default
        ``get_limits()``), excluding the major breaks. See ``minor_breaks``."""
        if limits is None:
            limits = self.get_limits()
        if limits is None or self.minor_breaks == ():
            return np.array([], dtype=float)
        limits = (float(min(limits)), float(max(limits)))
        major = self.get_breaks(limits)
        if self.minor_breaks is None:
            if isinstance(self.trans, LogTransform):
                return log_minor_breaks(major, limits, base=self.trans.base)
            return regular_minor_breaks(major, limits)
        if callable(self.minor_breaks):
            data = np.asarray(self.minor_breaks(self._data_limits(limits)), dtype=float).ravel()
        else:
            data = np.asarray(self.minor_breaks, dtype=float)
        t = self.trans.transform(data)
        t = t[np.isfinite(t) & np.isfinite(censor(t, limits))]
        if len(major) and len(t):
            tol = 1e-9 * max(limits[1] - limits[0], 1e-300)
            t = t[np.min(np.abs(t[:, None] - major[None, :]), axis=1) > tol]
        return np.unique(t)

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} aesthetics={self.aesthetics[0]!r} "
            f"trans={self.trans.name} limits={self.limits!r} name={self.name!r}>"
        )


_POSITIONS = {"x": ("bottom", "top"), "y": ("left", "right")}


class ScaleContinuousPosition(ScaleContinuous):
    """Continuous x or y position scale.

    ``position`` puts the axis (ticks, tick labels, title) on the given side of the
    panel: ``"bottom"`` (default) or ``"top"`` for x, ``"left"`` (default) or
    ``"right"`` for y; ``None`` means the default side. ``sec_axis`` (an
    :class:`~ggmanim.scales.sec_axis.AxisSecondary` from ``sec_axis()`` /
    ``dup_axis()``, or ``None``) adds a secondary axis on the opposite side.
    """

    def __init__(
        self, aesthetics, *args, position: str | None = None, sec_axis=None, **kwargs
    ) -> None:
        super().__init__(aesthetics, *args, **kwargs)
        if sec_axis is not None and not callable(getattr(sec_axis, "break_info", None)):
            raise TypeError(f"sec_axis must come from sec_axis() or dup_axis(), got {sec_axis!r}")
        self.sec_axis = sec_axis
        axis = "y" if self.aesthetic in Y_AESTHETICS else "x"
        if position is not None and position not in _POSITIONS[axis]:
            raise ValueError(
                f"position for an {axis} scale must be one of {_POSITIONS[axis]}, got {position!r}"
            )
        self.position = position if position is not None else _POSITIONS[axis][0]


# --------------------------------------------------------------------------- constructors
def _position(
    axis: str,
    trans,
    name,
    breaks,
    labels,
    limits,
    expand,
    oob,
    n_breaks,
    minor_breaks=None,
    position=None,
    sec_axis=None,
):
    aes = X_AESTHETICS if axis == "x" else Y_AESTHETICS
    return ScaleContinuousPosition(
        aes,
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
        sec_axis=sec_axis,
    )


@register
def scale_x_continuous(
    name=None,
    breaks=None,
    labels=None,
    limits=None,
    expand=None,
    oob="censor",
    trans="identity",
    n_breaks=5,
    minor_breaks=None,
    position=None,
    sec_axis=None,
):
    """Continuous x position scale. See :class:`ScaleContinuous` for arguments;
    ``position`` is "bottom"/"top" (see :class:`ScaleContinuousPosition`)."""
    return _position(
        "x",
        trans,
        name,
        breaks,
        labels,
        limits,
        expand,
        oob,
        n_breaks,
        minor_breaks,
        position,
        sec_axis,
    )


@register
def scale_y_continuous(
    name=None,
    breaks=None,
    labels=None,
    limits=None,
    expand=None,
    oob="censor",
    trans="identity",
    n_breaks=5,
    minor_breaks=None,
    position=None,
    sec_axis=None,
):
    """Continuous y position scale. See :class:`ScaleContinuous` for arguments;
    ``position`` is "left"/"right" (see :class:`ScaleContinuousPosition`)."""
    return _position(
        "y",
        trans,
        name,
        breaks,
        labels,
        limits,
        expand,
        oob,
        n_breaks,
        minor_breaks,
        position,
        sec_axis,
    )


@register
def scale_x_log10(
    name=None,
    breaks=None,
    labels=None,
    limits=None,
    expand=None,
    oob="censor",
    n_breaks=5,
    minor_breaks=None,
    position=None,
    sec_axis=None,
):
    """x scale with a log10 transform (breaks at powers of ten)."""
    return _position(
        "x",
        "log10",
        name,
        breaks,
        labels,
        limits,
        expand,
        oob,
        n_breaks,
        minor_breaks,
        position,
        sec_axis,
    )


@register
def scale_y_log10(
    name=None,
    breaks=None,
    labels=None,
    limits=None,
    expand=None,
    oob="censor",
    n_breaks=5,
    minor_breaks=None,
    position=None,
    sec_axis=None,
):
    """y scale with a log10 transform (breaks at powers of ten)."""
    return _position(
        "y",
        "log10",
        name,
        breaks,
        labels,
        limits,
        expand,
        oob,
        n_breaks,
        minor_breaks,
        position,
        sec_axis,
    )


@register
def scale_x_sqrt(
    name=None,
    breaks=None,
    labels=None,
    limits=None,
    expand=None,
    oob="censor",
    n_breaks=5,
    minor_breaks=None,
    position=None,
    sec_axis=None,
):
    """x scale with a square-root transform."""
    return _position(
        "x",
        "sqrt",
        name,
        breaks,
        labels,
        limits,
        expand,
        oob,
        n_breaks,
        minor_breaks,
        position,
        sec_axis,
    )


@register
def scale_y_sqrt(
    name=None,
    breaks=None,
    labels=None,
    limits=None,
    expand=None,
    oob="censor",
    n_breaks=5,
    minor_breaks=None,
    position=None,
    sec_axis=None,
):
    """y scale with a square-root transform."""
    return _position(
        "y",
        "sqrt",
        name,
        breaks,
        labels,
        limits,
        expand,
        oob,
        n_breaks,
        minor_breaks,
        position,
        sec_axis,
    )


@register
def scale_x_reverse(
    name=None,
    breaks=None,
    labels=None,
    limits=None,
    expand=None,
    oob="censor",
    n_breaks=5,
    minor_breaks=None,
    position=None,
    sec_axis=None,
):
    """Reversed x scale (large values on the left)."""
    return _position(
        "x",
        "reverse",
        name,
        breaks,
        labels,
        limits,
        expand,
        oob,
        n_breaks,
        minor_breaks,
        position,
        sec_axis,
    )


@register
def scale_y_reverse(
    name=None,
    breaks=None,
    labels=None,
    limits=None,
    expand=None,
    oob="censor",
    n_breaks=5,
    minor_breaks=None,
    position=None,
    sec_axis=None,
):
    """Reversed y scale (large values at the bottom)."""
    return _position(
        "y",
        "reverse",
        name,
        breaks,
        labels,
        limits,
        expand,
        oob,
        n_breaks,
        minor_breaks,
        position,
        sec_axis,
    )


def _lim(axis: str, args) -> ScaleContinuousPosition:
    if len(args) == 1 and np.ndim(args[0]) == 1:
        args = tuple(args[0])
    if len(args) != 2:
        raise ValueError(
            f"{axis}lim() takes exactly two values (lo, hi), got {len(args)}; "
            f"the old (min, max, step) form is replaced by scale_{axis}_continuous(breaks=...)"
        )
    lo, hi = args
    for v in (lo, hi):
        if v is not None and not isinstance(v, (int, float, np.integer, np.floating)):
            raise TypeError(f"{axis}lim() only supports numeric limits, got {v!r}")
    if lo is not None and hi is not None and lo > hi:
        # ggplot2: xlim(10, 0) gives a reversed scale
        return _position(axis, "reverse", None, None, None, (hi, lo), None, "censor", 5)
    return _position(axis, "identity", None, None, None, (lo, hi), None, "censor", 5)


@register
def xlim(*args) -> ScaleContinuousPosition:
    """Set x limits: ``xlim(lo, hi)`` or ``xlim((lo, hi))``. Data outside is removed
    (censored to NaN). ``lo > hi`` gives a reversed scale; either may be ``None``."""
    return _lim("x", args)


@register
def ylim(*args) -> ScaleContinuousPosition:
    """Set y limits: ``ylim(lo, hi)`` or ``ylim((lo, hi))``. See :func:`xlim`."""
    return _lim("y", args)
