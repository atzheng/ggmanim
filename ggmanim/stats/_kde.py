"""Kernel density estimation matching R's ``stats::density`` (numpy only).

:func:`density` reproduces ``density.default`` (R >= 4.4, ``old.coords = FALSE``):
the observations are linearly binned onto a grid of ``2 * n`` cells spanning
``[from - 4 bw, to + 4 bw]`` (R's ``BinDist``), convolved with the kernel by FFT,
then linearly interpolated onto ``n`` points from ``from`` to ``to``. This is the
estimate ggplot2's ``stat_density`` / ``stat_ydensity`` draw, so the curves agree
with R to rounding (``tests/fixtures/kde_r.R``).

:func:`compute_density` is ggplot2's ``compute_density``: normalised weights,
``bounds`` handled by reflection, and the computed variables ``density``,
``scaled``, ``ndensity``, ``count``, ``wdensity`` and ``n``.

Bandwidths (:func:`bandwidth`): ``"nrd0"`` (default, Silverman's rule of thumb,
alias ``"silverman"``), ``"nrd"`` (Scott's variation, alias ``"scott"``), a
positive number, or a callable ``f(x) -> bw``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = [
    "KERNELS",
    "bw_nrd0",
    "bw_nrd",
    "bw_ucv",
    "bw_bcv",
    "bw_sj",
    "bandwidth",
    "bin_dist",
    "density",
    "compute_density",
]

#: Kernels of R's ``density`` (each scaled so ``bw`` is its standard deviation).
KERNELS = (
    "gaussian",
    "epanechnikov",
    "rectangular",
    "triangular",
    "biweight",
    "cosine",
    "optcosine",
)

_KERNEL_ALIASES = {
    "gau": "gaussian",
    "epa": "epanechnikov",
    "rec": "rectangular",
    "tri": "triangular",
    "biw": "biweight",
    "cos": "cosine",
    "opt": "optcosine",
}


def _iqr(x: np.ndarray) -> float:
    q1, q3 = np.quantile(x, [0.25, 0.75])  # type 7, as R's IQR()
    return float(q3 - q1)


def bw_nrd0(x: Any) -> float:
    """R's ``bw.nrd0``: ``0.9 * min(sd, IQR / 1.34) * n ^ -0.2`` (with R's
    fallbacks when that minimum is 0)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        raise ValueError("need at least 2 data points to select a bandwidth")
    hi = float(np.std(x, ddof=1))
    lo = min(hi, _iqr(x) / 1.34)
    if not lo:
        lo = hi or abs(float(x[0])) or 1.0
    return 0.9 * lo * len(x) ** -0.2


def bw_nrd(x: Any) -> float:
    """R's ``bw.nrd`` (Scott's variation): ``1.06 * min(sd, IQR / 1.34) * n ^ -0.2``."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        raise ValueError("need at least 2 data points to select a bandwidth")
    return 1.06 * min(float(np.std(x, ddof=1)), _iqr(x) / 1.34) * len(x) ** -0.2


# -- cross-validation and plug-in bandwidths (R's bw.ucv, bw.bcv, bw.SJ) --------
# Ports of R's stats/R/bandwidths.R and src/library/stats/src/bandwidths.c,
# including R's optimize() (Brent's fmin) and uniroot() (R_zeroin2) so that the
# loose tolerances R uses give the same answer (tests/fixtures/bw_rules_r.R).

_DELMAX = 1000.0
_EPS = float(np.finfo(float).eps)


def _bw_pair_cnts(x: np.ndarray, nb: int) -> tuple[float, np.ndarray]:
    """R's ``bw_pair_cnts``: the bin width ``d`` and counts of pairwise bin
    distances (``cnt[k]`` = number of pairs ``k`` bins apart)."""
    n = len(x)
    lo, hi = float(x.min()), float(x.max())
    rang = (hi - lo) * 1.01
    if rang == 0:
        raise ValueError("data are constant in bandwidth calculation")
    d = rang / nb
    if n > nb / 2:  # binned (R >= 3.4): tabulate, then pair counts of the bins
        xx = np.trunc(np.abs(x) / d) * np.sign(x)
        xx = (xx - xx.min()).astype(np.int64)  # 0-based tabulate(xx - min + 1, nb)
        w = np.bincount(xx[xx < nb], minlength=nb)[:nb].astype(np.int64)
        cnt = np.correlate(w, w, mode="full")[nb - 1 :].astype(float)
        cnt[0] = float(np.sum(w * (w - 1))) / 2.0
        return d, cnt
    ii = np.trunc(x / d).astype(np.int64)  # C's (int) cast truncates
    i, j = np.triu_indices(n, k=1)
    cnt = np.bincount(np.abs(ii[i] - ii[j]), minlength=nb)[:nb].astype(float)
    return d, cnt


def _bin_sum(d: float, cnt: np.ndarray, h: float, term) -> float:
    """``sum(term(delta) * cnt)`` over bins with ``delta = (i d / h)^2 < 1000``,
    added left to right as R's C loop does."""
    delta = (np.arange(len(cnt)) * d / h) ** 2
    keep = delta < _DELMAX
    vals = term(delta[keep]) * cnt[keep]
    return float(np.cumsum(vals)[-1]) if len(vals) else 0.0


def _ucv(n: int, d: float, cnt: np.ndarray, h: float) -> float:
    s = _bin_sum(d, cnt, h, lambda t: np.exp(-t / 4.0) - np.sqrt(8.0) * np.exp(-t / 2.0))
    return (0.5 + s / n) / (n * h * np.sqrt(np.pi))


def _bcv(n: int, d: float, cnt: np.ndarray, h: float) -> float:
    s = _bin_sum(d, cnt, h, lambda t: np.exp(-t / 4.0) * (t * t - 12.0 * t + 12.0))
    return (1.0 + s / (32.0 * n)) / (2.0 * n * h * np.sqrt(np.pi))


def _phi4(n: int, d: float, cnt: np.ndarray, h: float) -> float:
    s = _bin_sum(d, cnt, h, lambda t: np.exp(-t / 2.0) * (t * t - 6.0 * t + 3.0))
    s = 2.0 * s + n * 3.0
    return s / (float(n) * (n - 1) * h**5.0 * np.sqrt(2 * np.pi))


def _phi6(n: int, d: float, cnt: np.ndarray, h: float) -> float:
    s = _bin_sum(d, cnt, h, lambda t: np.exp(-t / 2.0) * (t**3 - 15.0 * t * t + 45.0 * t - 15.0))
    s = 2.0 * s - 15.0 * n
    return s / (float(n) * (n - 1) * h**7.0 * np.sqrt(2 * np.pi))


def _fmin(f, ax: float, bx: float, tol: float) -> float:
    """R's ``optimize`` (``Brent_fmin``): a minimum of ``f`` on ``[ax, bx]``."""
    c = (3.0 - np.sqrt(5.0)) * 0.5
    eps = np.sqrt(_EPS)
    a, b = ax, bx
    v = w = x = a + c * (b - a)
    d = e = 0.0
    fx = fv = fw = f(x)
    tol3 = tol / 3.0
    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            return x
        p = q = r = 0.0
        if abs(e) > tol1:  # fit a parabola
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if abs(p) >= abs(q * 0.5 * r) or p <= q * (a - x) or p >= q * (b - x):
            e = (b - x) if x < xm else (a - x)  # golden section
            d = c * e
        else:  # parabolic interpolation
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = -tol1 if x >= xm else tol1
        if abs(d) >= tol1:
            u = x + d
        elif d > 0.0:
            u = x + tol1
        else:
            u = x - tol1
        fu = f(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv, w, fw = w, fw, u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu


def _zeroin(f, ax: float, bx: float, tol: float, maxit: int = 1000) -> float:
    """R's ``uniroot`` (``R_zeroin2``): a root of ``f`` in ``[ax, bx]``."""
    a, b = ax, bx
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    c, fc = a, fa
    for _ in range(maxit + 1):
        prev_step = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol_act = 2 * _EPS * abs(b) + tol / 2
        new_step = (c - b) / 2
        if abs(new_step) <= tol_act or fb == 0.0:
            return b
        if abs(prev_step) >= tol_act and abs(fa) > abs(fb):
            cb = c - b
            if a == c:  # linear interpolation
                t1 = fb / fa
                p = cb * t1
                q = 1.0 - t1
            else:  # inverse quadratic interpolation
                q, t1, t2 = fa / fc, fb / fc, fb / fa
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0))
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)
            if p > 0:
                q = -q
            else:
                p = -p
            if p < (0.75 * cb * q - abs(tol_act * q) / 2) and p < abs(prev_step * q / 2):
                new_step = p / q
        if abs(new_step) < tol_act:
            new_step = tol_act if new_step > 0 else -tol_act
        a, fa = b, fb
        b += new_step
        fb = f(b)
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c, fc = a, fa
    warnings.warn("bw_sj: root finding did not converge", stacklevel=3)
    return b


def _check_bw_input(x: Any, nb: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        raise ValueError("need at least 2 data points")
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite x")
    if int(nb) <= 0:
        raise ValueError("invalid 'nb'")
    return x


def _cv_bandwidth(x: Any, nb: int, crit, name: str) -> float:
    x = _check_bw_input(x, nb)
    n = len(x)
    hmax = 1.144 * float(np.std(x, ddof=1)) * n ** (-1 / 5)
    lower, upper = 0.1 * hmax, hmax
    tol = 0.1 * lower
    d, cnt = _bw_pair_cnts(x, int(nb))
    h = _fmin(lambda h: crit(n, d, cnt, h), lower, upper, tol)
    if h < lower + tol or h > upper - tol:
        warnings.warn(f"{name}: minimum occurred at one end of the range", stacklevel=3)
    return float(h)


def bw_ucv(x: Any, nb: int = 1000) -> float:
    """R's ``bw.ucv``: unbiased (least-squares) cross-validation bandwidth for a
    Gaussian kernel, minimised over ``[0.1 hmax, hmax]`` with
    ``hmax = 1.144 sd n^-1/5`` (pairwise distances binned into ``nb`` bins)."""
    return _cv_bandwidth(x, nb, _ucv, "bw_ucv")


def bw_bcv(x: Any, nb: int = 1000) -> float:
    """R's ``bw.bcv``: biased cross-validation bandwidth (see :func:`bw_ucv`)."""
    return _cv_bandwidth(x, nb, _bcv, "bw_bcv")


def bw_sj(x: Any, nb: int = 1000, method: str = "ste") -> float:
    """R's ``bw.SJ``: the Sheather & Jones (1991) plug-in bandwidth, by solving
    the equation (``method="ste"``, default) or direct plug-in (``"dpi"``)."""
    if method not in ("ste", "dpi"):
        raise ValueError(f"method must be 'ste' or 'dpi', got {method!r}")
    x = _check_bw_input(x, nb)
    n = len(x)
    d, cnt = _bw_pair_cnts(x, int(nb))
    q1, q3 = np.quantile(x, [0.25, 0.75])
    scale = min(float(np.std(x, ddof=1)), float(q3 - q1) / 1.349)
    a = 1.24 * scale * n ** (-1 / 7)
    b = 1.23 * scale * n ** (-1 / 9)
    c1 = 1 / (2 * np.sqrt(np.pi) * n)
    td = -_phi6(n, d, cnt, b)
    if not np.isfinite(td) or td <= 0:
        raise ValueError("sample is too sparse to find TD")
    if method == "dpi":
        return float((c1 / _phi4(n, d, cnt, (2.394 / (n * td)) ** (1 / 7))) ** (1 / 5))
    hmax = 1.144 * scale * n ** (-1 / 5)
    lower, upper = 0.1 * hmax, hmax
    tol = 0.1 * lower
    alph2 = 1.357 * (_phi4(n, d, cnt, a) / td) ** (1 / 7)
    if not np.isfinite(alph2):
        raise ValueError("sample is too sparse to find alph2")

    def fsd(h: float) -> float:
        return (c1 / _phi4(n, d, cnt, alph2 * h ** (5 / 7))) ** (1 / 5) - h

    itry = 1
    while fsd(lower) * fsd(upper) > 0:
        if itry > 99:
            raise ValueError("no solution in the specified range of bandwidths")
        if itry % 2:
            upper *= 1.2
        else:
            lower /= 1.2
        itry += 1
    return float(_zeroin(fsd, lower, upper, tol))


_BW_RULES: dict[str, Callable[[np.ndarray], float]] = {
    "nrd0": bw_nrd0,
    "silverman": bw_nrd0,
    "nrd": bw_nrd,
    "scott": bw_nrd,
    "ucv": bw_ucv,
    "bcv": bw_bcv,
    "sj": bw_sj,
    "sj-ste": bw_sj,
    "sj-dpi": lambda x: bw_sj(x, method="dpi"),
}


def bandwidth(x: Any, bw: Any = "nrd0") -> float:
    """The bandwidth for ``x``: a rule name (``"nrd0"``, ``"nrd"``, aliases
    ``"silverman"``/``"scott"``), a callable ``bw(x)`` or a positive number."""
    if isinstance(bw, str):
        rule = _BW_RULES.get(bw.lower())
        if rule is None:
            raise ValueError(
                f"unknown bandwidth rule {bw!r}; available: {', '.join(sorted(_BW_RULES))} "
                "(or a number / callable)"
            )
        value = rule(np.asarray(x, dtype=float))
    elif callable(bw):
        value = float(bw(np.asarray(x, dtype=float)))
    else:
        value = float(bw)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"bw must be a finite, positive number, got {value!r}")
    return value


def _kernel(name: str) -> str:
    key = str(name).lower()
    key = _KERNEL_ALIASES.get(key[:3], key) if key not in KERNELS else key
    if key not in KERNELS:
        raise ValueError(f"unknown kernel {name!r}; available: {', '.join(KERNELS)}")
    return key


def _kernel_values(kernel: str, u: np.ndarray, bw: float) -> np.ndarray:
    """Kernel ``kernel`` with standard deviation ``bw`` evaluated at ``u``."""
    au = np.abs(u)
    if kernel == "gaussian":
        return np.exp(-0.5 * (u / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
    if kernel == "rectangular":
        a = bw * np.sqrt(3)
        return np.where(au < a, 0.5 / a, 0.0)
    if kernel == "triangular":
        a = bw * np.sqrt(6)
        return np.where(au < a, (1 - au / a) / a, 0.0)
    if kernel == "epanechnikov":
        a = bw * np.sqrt(5)
        return np.where(au < a, 0.75 * (1 - (au / a) ** 2) / a, 0.0)
    if kernel == "biweight":
        a = bw * np.sqrt(7)
        return np.where(au < a, 15 / 16 * (1 - (au / a) ** 2) ** 2 / a, 0.0)
    if kernel == "cosine":
        a = bw / np.sqrt(1 / 3 - 2 / np.pi**2)
        return np.where(au < a, (1 + np.cos(np.pi * u / a)) / (2 * a), 0.0)
    a = bw / np.sqrt(1 - 8 / np.pi**2)  # optcosine
    return np.where(au < a, np.pi / 4 * np.cos(np.pi * u / (2 * a)) / a, 0.0)


def bin_dist(x: np.ndarray, w: np.ndarray, lo: float, hi: float, n: int) -> np.ndarray:
    """R's ``BinDist``: linear binning of ``x`` (weights ``w``) onto ``n`` grid
    points from ``lo`` to ``hi``; returns ``2 * n`` cells (the upper half zero
    padding for the FFT)."""
    y = np.zeros(2 * n)
    ok = np.isfinite(x)
    x, w = x[ok], w[ok]
    delta = (hi - lo) / (n - 1)
    pos = (x - lo) / delta
    ix = np.floor(pos).astype(np.int64)
    fx = pos - ix
    inner = (ix >= 0) & (ix <= n - 2)
    np.add.at(y, ix[inner], w[inner] * (1 - fx[inner]))
    np.add.at(y, ix[inner] + 1, w[inner] * fx[inner])
    left = ix == -1
    y[0] += float(np.sum(w[left] * fx[left]))
    right = ix == n - 1
    y[n - 1] += float(np.sum(w[right] * (1 - fx[right])))
    return y


def density(
    x: Any,
    weights: Any = None,
    bw: float = 1.0,
    kernel: str = "gaussian",
    n: int = 512,
    from_: float | None = None,
    to: float | None = None,
    cut: float = 3.0,
    ext: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """R's ``density(x, weights, bw, kernel = , n = , from = , to = )`` with a
    numeric ``bw`` (already multiplied by ``adjust``). Returns ``(grid, y)``.

    ``weights`` default to ``1 / len(x)``; non-finite ``x`` are dropped."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    big_n = len(x)
    if weights is None:
        xs = x[finite]
        w = np.full(len(xs), 1.0 / max(len(xs), 1))
        tot_mass = len(xs) / big_n if big_n else 1.0
    else:
        w_all = np.asarray(weights, dtype=float)
        wsum = float(w_all.sum())
        xs, w = x[finite], w_all[finite]
        tot_mass = float(w.sum()) / wsum if not finite.all() and wsum else 1.0
    kernel = _kernel(kernel)
    bw = float(bw)
    if not np.isfinite(bw) or bw <= 0:
        raise ValueError(f"bw must be a finite, positive number, got {bw!r}")
    n_user = int(n)
    m = max(n_user, 512)
    if m > 512:
        m = int(2 ** np.ceil(np.log2(m)))
    if from_ is None:
        from_ = float(xs.min()) - cut * bw
    if to is None:
        to = float(xs.max()) + cut * bw
    lo, up = from_ - ext * bw, to + ext * bw
    y = bin_dist(xs, w, lo, up, m) * tot_mass
    kords = np.linspace(0.0, (2 * m - 1) / (m - 1) * (up - lo), 2 * m)
    kords[m + 1 :] = -kords[1:m][::-1]
    kords = _kernel_values(kernel, kords, bw)
    conv = np.fft.ifft(np.fft.fft(y) * np.conj(np.fft.fft(kords)))
    dens = np.maximum(0.0, conv.real[:m])
    xords = np.linspace(lo, up, m)
    grid = np.linspace(from_, to, n_user)
    return grid, np.interp(grid, xords, dens)


def _reflect(
    grid: np.ndarray, dens: np.ndarray, bounds: tuple[float, float], from_: float, to: float
) -> tuple[np.ndarray, np.ndarray]:
    """ggplot2's ``reflect_density``: fold the density mass beyond finite
    ``bounds`` back inside."""

    def f(v: np.ndarray) -> np.ndarray:
        return np.interp(v, grid, dens, left=0.0, right=0.0)

    left, right = max(from_, bounds[0]), min(to, bounds[1])
    out = np.linspace(left, right, len(grid))
    y = f(out)
    with np.errstate(invalid="ignore"):
        if np.isfinite(bounds[0]):
            y = y + f(bounds[0] + (bounds[0] - out))
        if np.isfinite(bounds[1]):
            y = y + f(bounds[1] + (bounds[1] - out))
    return out, y


def compute_density(
    x: Any,
    w: Any = None,
    from_: float | None = None,
    to: float | None = None,
    bw: Any = "nrd0",
    adjust: float = 1.0,
    kernel: str = "gaussian",
    n: int = 512,
    bounds: tuple[float, float] = (-np.inf, np.inf),
) -> dict[str, np.ndarray] | None:
    """ggplot2's ``compute_density``: the density of ``x`` (weights ``w``,
    normalised to sum 1) on ``n`` points from ``from_`` to ``to`` (default: the
    data range), reflected at finite ``bounds``.

    Returns ``{"x", "density", "scaled", "ndensity", "count", "wdensity", "n"}``
    (arrays of equal length), or ``None`` for fewer than two observations."""
    x = np.asarray(x, dtype=float)
    nx = w_sum = float(len(x))
    if w is None:
        w = np.full(len(x), 1.0 / max(len(x), 1))
    else:
        w = np.asarray(w, dtype=float)
        w_sum = float(w.sum())
        w = w / w_sum if w_sum else w
    lo_b, hi_b = sorted(float(b) for b in bounds)
    inside = (lo_b <= x) & (x <= hi_b)
    fit_sum = 1.0
    if not inside.all():
        warnings.warn("Some data points are outside of `bounds`. Removing them.", stacklevel=3)
        x, w = x[inside], w[inside]
        fit_sum = float(w.sum())
        if fit_sum > 0:
            w = w / fit_sum
    w_sum = fit_sum * w_sum
    nx = float(len(x))
    if len(x) < 2:
        warnings.warn("Groups with fewer than two data points have been dropped.", stacklevel=3)
        return None
    if from_ is None:
        from_ = float(np.min(x))
    if to is None:
        to = float(np.max(x))
    h = bandwidth(x, bw) * float(adjust)
    if np.isfinite(lo_b) or np.isfinite(hi_b):
        width = to - from_
        lo = from_ - width * float(np.isfinite(lo_b))
        hi = to + width * float(np.isfinite(hi_b))
        m = int(n) * (int(np.isfinite(lo_b)) + int(np.isfinite(hi_b)) + 1)
        grid, dens = density(x, w, h, kernel, m, lo, hi)
        grid, dens = _reflect(grid, dens, (lo_b, hi_b), lo, hi)
    else:
        grid, dens = density(x, w, h, kernel, int(n), from_, to)
    peak = float(np.nanmax(dens)) if len(dens) else np.nan
    return {
        "x": grid,
        "density": dens,
        "scaled": dens / peak,
        "ndensity": dens / peak,
        "count": dens * nx,
        "wdensity": dens * w_sum,
        "n": np.full(len(grid), int(nx)),
    }
