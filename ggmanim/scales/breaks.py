"""Break computation, label formatting and range expansion helpers.

Pure numpy; no manim imports.

* :func:`pretty` -- port of R's ``pretty()`` (``R_pretty`` in ``src/appl/pretty.c``):
  1-2-5 "nice" steps, snapping outward to cover the range.
* :func:`log_breaks` -- port of ``scales::log_breaks()``: integer powers of the base,
  with 1/2/5-style sub-breaks when the range covers too few decades.
* :func:`extended` -- Wilkinson's extended labelling algorithm (Talbot, Lin & Hanrahan
  2010), ported from R's ``labeling::extended`` (ggplot2's default via
  ``scales::extended_breaks``).
* :func:`width_breaks` -- ``scales::fullseq``-style breaks every ``width``.
* :func:`regular_minor_breaks` / :func:`log_minor_breaks` -- minor breaks between majors.
* :func:`format_numbers` -- default tick-label formatter.
* :func:`expansion` / :func:`expand_range` -- ggplot2-style limit expansion.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence

import numpy as np

__all__ = [
    "pretty",
    "log_breaks",
    "extended",
    "width_breaks",
    "regular_minor_breaks",
    "log_minor_breaks",
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
    while by >= 1:  # scales: decrease ``by`` before resorting to sub-breaks
        powers = np.arange(pmin, pmax + 1, by, dtype=float)
        breaks = float(base) ** powers
        if n_inside(breaks) >= n - 2:
            return breaks
        by -= 1

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


# --------------------------------------------------------------------------- extended
_EXT_Q = (1, 5, 2, 2.5, 4, 3)
_EXT_W = (0.25, 0.2, 0.5, 0.05)


def _ext_simplicity(i: int, n: int, j: int, lmin: float, lmax: float, lstep: float) -> float:
    eps = _DBL_EPSILON * 100
    rem = lmin % lstep  # floored modulo, like R's %%
    v = 1 if ((rem < eps or lstep - rem < eps) and lmin <= 0 and lmax >= 0) else 0
    return 1 - (i - 1) / (n - 1) - j + v


def _ext_simplicity_max(i: int, n: int, j: int) -> float:
    return 1 - (i - 1) / (n - 1) - j + 1


def _ext_coverage(dmin: float, dmax: float, lmin: float, lmax: float) -> float:
    r = dmax - dmin
    return 1 - 0.5 * ((dmax - lmax) ** 2 + (dmin - lmin) ** 2) / ((0.1 * r) ** 2)


def _ext_coverage_max(dmin: float, dmax: float, span: float) -> float:
    r = dmax - dmin
    if span > r:
        half = (span - r) / 2
        return 1 - 0.5 * (2 * half**2) / ((0.1 * r) ** 2)
    return 1.0


def _ext_density(k: int, m: int, dmin: float, dmax: float, lmin: float, lmax: float) -> float:
    r = (k - 1) / (lmax - lmin)
    rt = (m - 1) / (max(lmax, dmax) - min(dmin, lmin))
    return 2 - max(r / rt, rt / r)


def _ext_density_max(k: int, m: int) -> float:
    return 2 - (k - 1) / (m - 1) if k >= m else 1.0


def extended(
    dmin: float,
    dmax: float,
    m: int = 5,
    Q: Sequence[float] = _EXT_Q,
    only_loose: bool = False,
    w: Sequence[float] = _EXT_W,
) -> np.ndarray:
    """Wilkinson's *extended* labelling algorithm (``labeling::extended``).

    Searches step sizes ``j * q * 10^z`` (``q`` from the "nice" list ``Q``, in order
    of preference) and label counts ``k`` near ``m``, maximising a weighted sum of
    simplicity, coverage, density and legibility (weights ``w``). Unlike
    :func:`pretty`, the labels need not cover the data range (unless
    ``only_loose``); e.g. ``extended(1, 10)`` gives ``0, 2.5, 5, 7.5, 10``.

    >>> extended(0, 100).tolist()
    [0.0, 25.0, 50.0, 75.0, 100.0]
    """
    dmin, dmax = float(dmin), float(dmax)
    if not (math.isfinite(dmin) and math.isfinite(dmax)):
        raise ValueError(f"extended() needs finite bounds, got ({dmin}, {dmax})")
    if dmin > dmax:
        dmin, dmax = dmax, dmin
    m = int(m)
    if m < 2:
        raise ValueError(f"extended() needs m >= 2, got {m}")
    if dmax - dmin < _DBL_EPSILON * 100:
        return np.linspace(dmin, dmax, m)
    Q = tuple(float(q) for q in Q)
    n = len(Q)
    w1, w2, w3, w4 = (float(v) for v in w)
    best_score = -2.0
    best: tuple[float, float, float] | None = None

    j = 1
    done = False
    while not done:
        for i, q in enumerate(Q, start=1):
            sm = _ext_simplicity_max(i, n, j)
            if w1 * sm + w2 + w3 + w4 < best_score:
                done = True
                break
            k = 2
            while True:
                dm = _ext_density_max(k, m)
                if w1 * sm + w2 + w3 * dm + w4 < best_score:
                    break
                delta = (dmax - dmin) / (k + 1) / j / q
                z = math.ceil(math.log10(delta))
                while True:
                    step = j * q * 10.0**z
                    cm = _ext_coverage_max(dmin, dmax, step * (k - 1))
                    if w1 * sm + w2 * cm + w3 * dm + w4 < best_score:
                        break
                    min_start = math.floor(dmax / step) * j - (k - 1) * j
                    max_start = math.ceil(dmin / step) * j
                    for start in range(min_start, max_start + 1):
                        lmin = start * (step / j)
                        lmax = lmin + step * (k - 1)
                        s = _ext_simplicity(i, n, j, lmin, lmax, step)
                        c = _ext_coverage(dmin, dmax, lmin, lmax)
                        g = _ext_density(k, m, dmin, dmax, lmin, lmax)
                        score = w1 * s + w2 * c + w3 * g + w4 * 1.0  # legibility == 1
                        if score > best_score and (
                            not only_loose or (lmin <= dmin and lmax >= dmax)
                        ):
                            best_score = score
                            best = (lmin, lmax, step)
                    z += 1
                k += 1
        j += 1
    assert best is not None
    lmin, lmax, step = best
    count = int(round((lmax - lmin) / step)) + 1
    out = lmin + step * np.arange(count, dtype=float)
    return np.round(out, max(0, 2 - math.floor(math.log10(step)))) + 0.0


def width_breaks(lo: float, hi: float, width: float, offset: float = 0.0) -> np.ndarray:
    """Breaks every ``width`` covering ``[lo, hi]`` (``scales::fullseq``), shifted by
    ``offset``: ``floor(lo / width) * width, ..., ceil(hi / width) * width``."""
    lo, hi = sorted((float(lo), float(hi)))
    width = float(width)
    if not (width > 0 and math.isfinite(width)):
        raise ValueError(f"width must be positive and finite, got {width}")
    start = math.floor(lo / width + 1e-10)
    stop = math.ceil(hi / width - 1e-10)
    if stop <= start:
        stop = start + 1
    out = np.arange(start, stop + 1, dtype=float) * width + float(offset)
    return _round_sig(out)


def regular_minor_breaks(
    major: Sequence[float] | np.ndarray, limits: Sequence[float], n: int = 2
) -> np.ndarray:
    """Minor breaks dividing each gap between (equally spaced) ``major`` breaks into
    ``n`` parts (``scales::regular_minor_breaks``), extended one major step beyond
    the first/last major where the ``limits`` reach. Returned *without* the majors,
    censored to ``limits``; all values in the same (transformed) space."""
    b = np.asarray(major, dtype=float).ravel()
    b = np.sort(b[np.isfinite(b)])
    if len(b) < 2 or n < 2:
        return np.array([], dtype=float)
    lo, hi = float(min(limits)), float(max(limits))
    bd = b[1] - b[0]
    if lo < b[0]:
        b = np.concatenate([[b[0] - bd], b])
    if hi > b[-1]:
        b = np.concatenate([b, [b[-1] + bd]])
    frac = np.arange(1, n) / n
    out = (b[:-1, None] + np.diff(b)[:, None] * frac[None, :]).ravel()
    return _censor_minor(out, b, lo, hi)


def log_minor_breaks(
    major: Sequence[float] | np.ndarray, limits: Sequence[float], base: float = 10
) -> np.ndarray:
    """Minor breaks for a log scale, in *transformed* (log) space: ``k * base^p`` for
    ``k = 2 .. base - 1`` and every power ``p`` the ``limits`` touch, excluding the
    ``major`` breaks and censored to ``limits`` (both in log space)."""
    lo, hi = float(min(limits)), float(max(limits))
    if not (math.isfinite(lo) and math.isfinite(hi)) or base <= 2:
        return np.array([], dtype=float)
    powers = np.arange(math.floor(lo) - 1, math.ceil(hi) + 1, dtype=float)
    ks = np.arange(2, int(math.ceil(base)), dtype=float)
    data = np.outer(float(base) ** powers, ks).ravel()
    out = np.log(data) / math.log(base)
    return _censor_minor(out, np.asarray(major, dtype=float), lo, hi)


def _censor_minor(x: np.ndarray, major: np.ndarray, lo: float, hi: float) -> np.ndarray:
    tol = 1e-9 * max(hi - lo, 1e-300)
    x = x[(x >= lo - tol) & (x <= hi + tol)]
    major = major[np.isfinite(major)]
    if len(major):
        x = x[np.min(np.abs(x[:, None] - major[None, :]), axis=1) > tol]
    return np.unique(x)


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


#: Significant digits shown by the default formatters (R's ``getOption("digits")``).
_DISPLAY_DIGITS = 7


def _display_decimals(x: np.ndarray, digits: int = _DISPLAY_DIGITS, max_digits: int = 15) -> int:
    """Common number of decimals for labelling ``x``: the fewest that show every
    value exactly, but never more than needed to show each value to ``digits``
    significant figures (like R's ``format()``), so unrounded values such as
    ``0.9999999999`` print as ``1`` instead of ``1.000000000``."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return 0
    rounded = _round_sig(x, digits)
    nz = rounded != 0
    exps = np.floor(np.log10(np.abs(np.where(nz, rounded, 1.0))) + 1e-12)
    caps = np.where(nz, np.clip(digits - 1 - exps, 0, max_digits), 0).astype(int)
    tol = 1e-10 * max(float(np.max(np.abs(x))), 1e-300)
    d = 0
    for v, cap in zip(rounded, caps):
        # fewest decimals showing v exactly, capped at its last significant digit;
        # integer digits are never dropped
        need = next((k for k in range(cap + 1) if abs(round(v, k) - v) <= tol), cap)
        d = max(d, need)
    return d


def format_numbers(values) -> list[str]:
    """Default tick-label formatter.

    All labels share the same number of decimals -- the fewest needed to show every
    value (so trailing zeros are trimmed, but ``0.0, 0.5, 1.0`` stay aligned, as in
    ggplot2). Scientific notation (R style, ``"1e+05"``) is used when any non-zero
    value has a decimal exponent with ``|exp| >= 5``. Values are shown to at most
    7 significant digits (R's default ``digits``), so unrounded limits don't print
    as ``"1.000000000"``. Non-finite values give ``""``.
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
        exps_all = np.where(
            xf != 0, np.floor(np.log10(np.abs(np.where(xf != 0, xf, 1))) + 1e-12), 0
        )
        mant = xf / 10.0**exps_all
        d = _display_decimals(mant, max_digits=14)
        labels = []
        for m, e in zip(mant, exps_all):
            if round(abs(m), d) >= 10:  # rounding pushed mantissa to 10
                m, e = m / 10, e + 1
            sign = "-" if e < 0 else "+"
            labels.append(f"{m:.{d}f}e{sign}{int(abs(e)):02d}")
    else:
        d = _display_decimals(xf)
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
