"""Break computation, label formatting and range expansion helpers.

Pure numpy; no manim imports.

* :func:`pretty` -- port of R's ``pretty()`` (``R_pretty`` in ``src/appl/pretty.c``):
  1-2-5 "nice" steps, snapping outward to cover the range.
* :func:`log_breaks` -- port of ``scales::log_breaks()``: integer powers of the base,
  with 1/2/5-style sub-breaks when the range covers too few decades.
* :func:`format_numbers` -- default tick-label formatter.
* :func:`expansion` / :func:`expand_range` -- ggplot2-style limit expansion.
"""

from __future__ import annotations

import math
import sys
from typing import Sequence

import numpy as np

__all__ = [
    "pretty",
    "log_breaks",
    "format_numbers",
    "expansion",
    "expand_range",
    "DEFAULT_CONTINUOUS_EXPANSION",
]

_DBL_EPSILON = sys.float_info.epsilon
_DBL_MIN = sys.float_info.min


def pretty(lo: float, hi: float, n: int = 5, min_n: int | None = None) -> np.ndarray:
    """Equally spaced "nice" breaks covering ``[lo, hi]`` (R's ``pretty()``).

    Steps are 1, 2, 5 or 10 times a power of ten; the first break is <= ``lo`` and
    the last >= ``hi``. ``n`` is the desired number of intervals (not a guarantee).

    >>> pretty(0, 1).tolist()
    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    """
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"pretty() needs finite bounds, got ({lo}, {hi})")
    if lo > hi:
        lo, hi = hi, lo
    ndiv = int(n)
    if min_n is None:
        min_n = ndiv // 3
    shrink_sml = 0.75
    h = 1.5
    h5 = 0.5 + 1.5 * h
    rounding_eps = 1e-10

    dx = hi - lo
    if dx == 0 and hi == 0:
        cell = 1.0
        i_small = True
    else:
        cell = max(abs(lo), abs(hi))
        u = 1 + (1 / (1 + h) if h5 >= 1.5 * h + 0.5 else 1.5 / (1 + h5))
        u *= max(1, ndiv) * _DBL_EPSILON
        i_small = dx < cell * u * 3

    if i_small:
        if cell > 10:
            cell = 9 + cell / 10
        cell *= shrink_sml
        if min_n > 1:
            cell /= min_n
    else:
        cell = dx
        if ndiv > 1:
            cell /= ndiv

    if cell < 20 * _DBL_MIN:
        cell = 20 * _DBL_MIN
    elif cell * 10 > sys.float_info.max:
        cell = 0.1 * sys.float_info.max

    base = 10.0 ** math.floor(math.log10(cell))
    unit = base
    ns = 2 * base
    if ns - cell < h * (cell - unit):
        unit = ns
        ns = 5 * base
        if ns - cell < h5 * (cell - unit):
            unit = ns
            ns = 10 * base
            if ns - cell < h * (cell - unit):
                unit = ns

    ns = math.floor(lo / unit + rounding_eps)
    nu = math.ceil(hi / unit - rounding_eps)
    while ns * unit > lo + rounding_eps * unit:
        ns -= 1
    while nu * unit < hi - rounding_eps * unit:
        nu += 1

    k = int(0.5 + nu - ns)
    if k < min_n:
        k = min_n - k
        if ns >= 0:
            nu += k // 2
            ns -= k // 2 + k % 2
        else:
            ns -= k // 2
            nu += k // 2 + k % 2

    out = np.arange(ns, nu + 1, dtype=float) * unit
    # remove floating-point noise such as 0.6000000000000001
    return np.round(out, max(0, 2 - math.floor(math.log10(unit)))) + 0.0


def log_breaks(lo: float, hi: float, n: int = 5, base: float = 10) -> np.ndarray:
    """Breaks for a log-transformed scale, in *data* space (``scales::log_breaks``).

    Uses integer powers of ``base``; if fewer than ``n - 2`` of them fall inside
    ``[lo, hi]`` it adds sub-breaks (1,5 then 1,2,5 then 1..9 times each power), and
    finally falls back to :func:`pretty` on the raw range. ``lo``/``hi`` must be > 0.
    The result may extend beyond ``[lo, hi]``; callers censor to the limits.
    """
    lo, hi = sorted((float(lo), float(hi)))
    if not (lo > 0 and math.isfinite(hi)):
        raise ValueError(f"log_breaks() needs a positive finite range, got ({lo}, {hi})")
    rlo = math.log(lo, base)
    rhi = math.log(hi, base)
    pmin = math.floor(rlo + 1e-10)
    pmax = math.ceil(rhi - 1e-10)
    if pmax == pmin:
        return np.array([float(base) ** pmin])

    def n_inside(b: np.ndarray) -> int:
        tol = 1e-10 * max(abs(lo), abs(hi))
        return int(np.sum((b >= lo - tol) & (b <= hi + tol)))

    by = (pmax - pmin) // n + 1
    powers = np.arange(pmin, pmax + 1, by, dtype=float)
    breaks = float(base) ** powers
    if n_inside(breaks) >= n - 2:
        return breaks

    if base == 10:
        candidates: list[Sequence[int]] = [(1, 5), (1, 2, 5), tuple(range(1, 10))]
    else:
        candidates = [tuple(range(1, int(base)))] if base > 2 else []
    powers = np.arange(pmin - 1, pmax + 1, dtype=float)
    for steps in candidates:
        sub = np.unique(_round_sig(np.outer(float(base) ** powers, steps).ravel()))
        if n_inside(sub) >= n - 2:
            return sub
    return pretty(lo, hi, n)


def _round_sig(x: np.ndarray, digits: int = 12) -> np.ndarray:
    """Round to ``digits`` significant figures (removes float noise like 0.30000000000000004)."""
    return np.array([float(f"{v:.{digits}g}") for v in np.asarray(x, dtype=float).ravel()])


def _common_decimals(x: np.ndarray, max_digits: int = 15) -> int:
    """Smallest number of decimals that represents every value in ``x`` exactly
    (to ~10 significant digits)."""
    tol = 1e-10 * max(float(np.max(np.abs(x))), 1e-300)
    for d in range(max_digits + 1):
        if np.all(np.abs(np.round(x, d) - x) <= tol):
            return d
    return max_digits


def format_numbers(values) -> list[str]:
    """Default tick-label formatter.

    All labels share the same number of decimals -- the fewest needed to show every
    value (so trailing zeros are trimmed, but ``0.0, 0.5, 1.0`` stay aligned, as in
    ggplot2). Scientific notation (R style, ``"1e+05"``) is used when any non-zero
    value has a decimal exponent with ``|exp| >= 5``. Non-finite values give ``""``.
    """
    x = np.asarray(values, dtype=float).ravel()
    out = [""] * len(x)
    finite = np.isfinite(x)
    if not finite.any():
        return out
    xf = x[finite] + 0.0  # turn -0.0 into 0.0
    nz = xf[xf != 0]
    sci = False
    if len(nz):
        exps = np.floor(np.log10(np.abs(nz)) + 1e-12)
        sci = bool(np.any(np.abs(exps) >= 5))
    if sci:
        exps_all = np.where(xf != 0, np.floor(np.log10(np.abs(np.where(xf != 0, xf, 1))) + 1e-12), 0)
        mant = xf / 10.0**exps_all
        d = _common_decimals(np.round(mant, 12), max_digits=14)
        labels = []
        for m, e in zip(mant, exps_all):
            if round(abs(m), d) >= 10:  # rounding pushed mantissa to 10
                m, e = m / 10, e + 1
            sign = "-" if e < 0 else "+"
            labels.append(f"{m:.{d}f}e{sign}{int(abs(e)):02d}")
    else:
        d = _common_decimals(xf)
        labels = [f"{v:.{d}f}" for v in np.round(xf, d) + 0.0]
    it = iter(labels)
    for i, ok in enumerate(finite):
        if ok:
            out[i] = next(it)
    return out


#: ggplot2's default for continuous scales: 5% of the range on each side.
DEFAULT_CONTINUOUS_EXPANSION = (0.05, 0.0, 0.05, 0.0)


def expansion(mult=0.0, add=0.0) -> tuple[float, float, float, float]:
    """ggplot2's ``expansion()``: build an expansion vector ``(mult_lo, add_lo,
    mult_hi, add_hi)``. ``mult`` and ``add`` may each be a scalar (both sides) or a
    ``(lower, upper)`` pair. ``add`` is in transformed-scale units."""
    m = np.broadcast_to(np.asarray(mult, dtype=float), (2,))
    a = np.broadcast_to(np.asarray(add, dtype=float), (2,))
    return (float(m[0]), float(a[0]), float(m[1]), float(a[1]))


def _normalise_expand(expand) -> tuple[float, float, float, float]:
    if expand is None or expand is True:
        return DEFAULT_CONTINUOUS_EXPANSION
    if expand is False:
        return (0.0, 0.0, 0.0, 0.0)
    e = tuple(float(v) for v in expand)
    if len(e) == 2:  # (mult, add), as in ggplot2's c(m, a)
        return (e[0], e[1], e[0], e[1])
    if len(e) == 4:
        return e  # type: ignore[return-value]
    raise ValueError(f"expand must have length 2 (mult, add) or 4, got {expand!r}")


def expand_range(limits, expand=None, zero_width: float = 1.0) -> tuple[float, float]:
    """Expand ``limits`` (``(lo, hi)``) by an expansion vector (see :func:`expansion`).

    ``expand`` may be ``None``/``True`` (default 5%), ``False`` (none), ``(mult, add)``
    or ``(mult_lo, add_lo, mult_hi, add_hi)``. A zero-width range is widened to
    ``zero_width`` centred on the value (like ``scales::expand_range``).
    """
    lo, hi = float(limits[0]), float(limits[1])
    m_lo, a_lo, m_hi, a_hi = _normalise_expand(expand)
    if hi == lo:
        return (lo - zero_width / 2, hi + zero_width / 2)
    width = hi - lo
    return (lo - width * m_lo - a_lo, hi + width * m_hi + a_hi)
