"""stat_density (for geom_density) and stat_ydensity (for geom_violin).

Both estimate densities with :func:`ggmanim.stats._kde.compute_density`, which
reproduces R's ``density()`` (Gaussian kernel by default, ``bw="nrd0"``,
``adjust``, ``kernel``, reflection at finite ``bounds``) and honour a ``weight``
aesthetic. Besides the rules of :func:`ggmanim.stats._kde.bandwidth`, ``bw``
accepts ggplot2's ``"ucv"``, ``"bcv"``, ``"SJ"`` (= ``"sj-ste"``) and
``"sj-dpi"`` (:func:`bw_ucv`, :func:`bw_bcv`, :func:`bw_sj`: ports of R's
``bw.ucv``/``bw.bcv``/``bw.SJ``, computed on the unweighted values as in R).

``stat_density`` -- per group, ``n = 512`` points over the x scale's range (or
the group's own range with ``trim=True``). Computed variables: ``density``,
``count`` (``density * n``), ``wdensity`` (``density * sum(weight)``),
``scaled``/``ndensity`` (``density / max``) and ``n``. ``geom_density`` maps
``y = after_stat("density")`` by default. ``orientation="y"`` estimates the
density of ``y`` (and maps ``x = after_stat("density")``).

``stat_ydensity`` -- per group (and per x level on a discrete x scale) the
density of ``y`` over the group's range (``trim=True``) or that range extended by
three bandwidths (``trim=False``); ``x`` stays the group's position. Adds
``violinwidth``, the half-width of the violin relative to its ``width``:
``scale="area"`` (every violin has the same area: ``density / max(density)``
over the panel), ``"count"`` (areas proportional to the number of observations)
or ``"width"`` (every violin as wide as ``width``: ``scaled``). ``quantiles``
(default ``(0.25, 0.5, 0.75)``) are inserted as extra rows with a ``quantile``
column (``NaN`` on the other rows), which ``geom_violin(draw_quantiles=...)``
draws.

Horizontal violins (``y`` discrete) are computed on swapped columns, see
:func:`ggmanim.stats.boxplot.resolve_flipped`.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from ..aes import after_stat
from ..registry import register
from ._kde import bandwidth, compute_density
from .base import Stat, scale_limits
from .boxplot import _finite_y, _is_discrete_scale, flip_names, regroup_by_x, resolve_flipped

__all__ = [
    "StatDensity",
    "StatYdensity",
    "bw_bcv",
    "bw_sj",
    "bw_ucv",
    "density_mapping",
    "resolve_bw",
    "stat_density",
    "stat_ydensity",
]

_VIOLIN_SCALES = ("area", "count", "width")

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


#: ggplot2's ``precompute_bw`` rule names handled here (the others go to
#: :func:`ggmanim.stats._kde.bandwidth`).
_EXTRA_BW = {
    "ucv": bw_ucv,
    "bcv": bw_bcv,
    "sj": bw_sj,
    "sj-ste": bw_sj,
    "sj-dpi": lambda x: bw_sj(x, method="dpi"),
}


def resolve_bw(x: Any, bw: Any) -> Any:
    """``bw`` as accepted by :func:`ggmanim.stats._kde.bandwidth`: the rules
    ``"ucv"``, ``"bcv"``, ``"SJ"`` (= ``"sj-ste"``) and ``"sj-dpi"`` (any case,
    as ggplot2's ``precompute_bw``) are evaluated on ``x`` here; anything else
    is returned unchanged."""
    if isinstance(bw, str) and bw.lower() in _EXTRA_BW:
        return _EXTRA_BW[bw.lower()](np.asarray(x, dtype=float))
    return bw


def _weights(df: pd.DataFrame) -> np.ndarray | None:
    if "weight" not in df.columns:
        return None
    return pd.to_numeric(df["weight"], errors="coerce").to_numpy(dtype=float)


class StatDensity(Stat):
    """Kernel density estimate of ``x`` per group (see the module docstring)."""

    dropped_aes = ("weight",)

    def __init__(
        self,
        bw: Any = "nrd0",
        adjust: float = 1.0,
        kernel: str = "gaussian",
        n: int = 512,
        trim: bool = False,
        bounds: tuple[float, float] = (-np.inf, np.inf),
        orientation: str | None = None,
        na_rm: bool = False,
        **params,
    ):
        super().__init__(
            bw=bw,
            adjust=adjust,
            kernel=kernel,
            n=n,
            trim=trim,
            bounds=bounds,
            orientation=orientation,
            na_rm=na_rm,
            **params,
        )

    def compute_panel(self, df: pd.DataFrame, scales=None, **params) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        flipped = resolve_flipped(df, scales, params.get("orientation"), main_is_orthogonal=False)
        data = flip_names(df) if flipped else df.copy()
        if "x" not in data.columns:
            raise ValueError("stat_density requires an x aesthetic (or y with orientation='y')")
        data = data.drop(columns=["flipped_aes", "y"], errors="ignore")
        data = _finite_y(data.rename(columns={"x": "y"}), "stat_density", bool(params.get("na_rm")))
        data = data.rename(columns={"y": "x"})
        if data.empty:
            return data
        rng = scale_limits(scales, "y" if flipped else "x")
        if rng is None:
            x = pd.to_numeric(data["x"], errors="coerce").to_numpy(dtype=float)
            rng = (float(x.min()), float(x.max()))
        params = {**params, "flipped_aes": flipped, "range_": rng}
        out = super().compute_panel(data, scales, **params)
        return flip_names(out) if flipped else out

    def compute_group(  # type: ignore[override]
        self,
        df: pd.DataFrame,
        scales=None,
        bw: Any = "nrd0",
        adjust: float = 1.0,
        kernel: str = "gaussian",
        n: int = 512,
        trim: bool = False,
        bounds: tuple[float, float] = (-np.inf, np.inf),
        orientation: str | None = None,
        na_rm: bool = False,
        flipped_aes: bool = False,
        range_: tuple[float, float] | None = None,
    ) -> pd.DataFrame:
        x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
        if trim or range_ is None:
            lo, hi = float(x.min()), float(x.max())
        else:
            lo, hi = range_
        bw = resolve_bw(x, bw)
        dens = compute_density(x, _weights(df), lo, hi, bw, adjust, kernel, int(n), bounds)
        if dens is None:
            return pd.DataFrame()
        out = pd.DataFrame(dens)
        out["flipped_aes"] = flipped_aes
        return out


class StatYdensity(Stat):
    """Density of ``y`` per group, for violins (see the module docstring)."""

    dropped_aes = ("weight",)

    def __init__(
        self,
        bw: Any = "nrd0",
        adjust: float = 1.0,
        kernel: str = "gaussian",
        trim: bool = True,
        scale: str = "area",
        drop: bool = True,
        bounds: tuple[float, float] = (-np.inf, np.inf),
        quantiles: Any = (0.25, 0.5, 0.75),
        orientation: str | None = None,
        na_rm: bool = False,
        **params,
    ):
        if scale not in _VIOLIN_SCALES:
            raise ValueError(f"scale must be one of {_VIOLIN_SCALES}, got {scale!r}")
        super().__init__(
            bw=bw,
            adjust=adjust,
            kernel=kernel,
            trim=trim,
            scale=scale,
            drop=drop,
            bounds=bounds,
            quantiles=quantiles,
            orientation=orientation,
            na_rm=na_rm,
            **params,
        )

    def compute_panel(self, df: pd.DataFrame, scales=None, **params) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        flipped = resolve_flipped(df, scales, params.get("orientation"))
        data = flip_names(df) if flipped else df.copy()
        if "y" not in data.columns:
            raise ValueError("stat_ydensity requires a y aesthetic (or x with orientation='y')")
        if "x" not in data.columns:
            data["x"] = 0.0
        data = _finite_y(data, "stat_ydensity", bool(params.get("na_rm")))
        if data.empty:
            return data
        if _is_discrete_scale(scales, "y" if flipped else "x"):
            data = regroup_by_x(data)
        data = data.drop(columns=["flipped_aes"], errors="ignore")
        scale = params.pop("scale", "area")
        if scale not in _VIOLIN_SCALES:
            raise ValueError(f"scale must be one of {_VIOLIN_SCALES}, got {scale!r}")
        out = super().compute_panel(data, scales, **params)
        if len(out) and "density" in out.columns:
            dens = pd.to_numeric(out["density"], errors="coerce").to_numpy(dtype=float)
            peak = np.nanmax(dens)
            if scale == "area":
                vw = dens / peak
            elif scale == "count":
                nn = pd.to_numeric(out["n"], errors="coerce").to_numpy(dtype=float)
                vw = dens / peak * nn / np.nanmax(nn)
            else:
                vw = pd.to_numeric(out["scaled"], errors="coerce").to_numpy(dtype=float)
            out["violinwidth"] = vw
        out["flipped_aes"] = flipped
        return flip_names(out) if flipped else out

    @classmethod
    def parameters(cls) -> set:
        return super().parameters() | {"scale"}

    def compute_group(  # type: ignore[override]
        self,
        df: pd.DataFrame,
        scales=None,
        bw: Any = "nrd0",
        adjust: float = 1.0,
        kernel: str = "gaussian",
        trim: bool = True,
        drop: bool = True,
        bounds: tuple[float, float] = (-np.inf, np.inf),
        quantiles: Any = (0.25, 0.5, 0.75),
        orientation: str | None = None,
        na_rm: bool = False,
    ) -> pd.DataFrame:
        y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
        x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
        if len(y) < 2:
            if drop:
                warnings.warn(
                    "stat_ydensity: groups with fewer than two data points have been dropped",
                    stacklevel=4,
                )
                return pd.DataFrame()
            return pd.DataFrame({"x": x, "n": len(y)})
        h = bandwidth(y, resolve_bw(y, bw))
        mod = 0.0 if trim else 3.0
        lo, hi = float(y.min()) - mod * h, float(y.max()) + mod * h
        dens = compute_density(y, _weights(df), lo, hi, h, adjust, kernel, 512, bounds)
        if dens is None:
            return pd.DataFrame()
        out = pd.DataFrame(dens).rename(columns={"x": "y"})
        if len(np.unique(x)) > 1:
            out["x"] = (x.min() + x.max()) / 2
            out["width"] = float(np.ptp(x)) * 0.9
        else:
            out["x"] = x[0]
        if quantiles is not None and len(np.atleast_1d(quantiles)):
            probs = np.atleast_1d(np.asarray(quantiles, dtype=float))
            if ((probs < 0) | (probs > 1)).any():
                raise ValueError("quantiles must be between 0 and 1")
            qy = np.quantile(y, probs)
            grid = out["y"].to_numpy(dtype=float)
            q = pd.DataFrame({"y": qy, "quantile": probs})
            for col in out.columns:
                if col in q.columns:
                    continue
                v = out[col]
                if pd.api.types.is_numeric_dtype(v):
                    q[col] = np.interp(qy, grid, v.to_numpy(dtype=float))
                else:
                    q[col] = v.iloc[0]
            out = out[~np.isin(grid, qy)]
            out = pd.concat([out, q], ignore_index=True)
        return out


def _stat_ctor(stat: str, geom, mapping, data, position, inherit_aes, name, params):
    from ..geoms.count import stat_layer  # geoms import manim: keep stats free of it

    return stat_layer(
        stat,
        geom,
        mapping,
        data,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )


@register
def stat_density(
    mapping=None,
    data=None,
    *,
    geom="area",
    position="stack",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Kernel density estimate drawn with ``geom`` (default a stacked
    ``geom_area``, as in ggplot2); ``y = after_stat("density")`` unless mapped.
    Params: ``bw``, ``adjust``, ``kernel``, ``n``, ``trim``, ``bounds``. Returns a
    layer.

    The default ``y`` (or, for a horizontal density, ``x``) is added when the
    layer's mapping is merged with the plot's, so ``ggplot(df, aes(y="v")) +
    stat_density()`` is horizontal (see :func:`density_mapping`)."""
    from ..geoms.density import DensityLayer  # geoms import manim: keep stats free of it

    layer = _stat_ctor("density", geom, mapping, data, position, inherit_aes, name, params)
    return DensityLayer.from_layer(layer)


def density_mapping(mapping, params):
    """``mapping`` plus the density default ``y = after_stat("density")``, or
    ``x = after_stat("density")`` for a horizontal density: ``orientation="y"``,
    or no orientation and ``y`` (not computed) mapped without ``x``. Nothing is
    added when that aesthetic is already mapped in ``mapping`` or set as a
    constant in ``params``. Applied to the layer mapping merged with the plot
    mapping (:class:`ggmanim.geoms.density.DensityLayer`)."""
    from ..aes import Aes, standardise_aes_name

    if isinstance(mapping, pd.DataFrame):
        return mapping
    given = Aes(mapping) if mapping is not None else Aes()
    orientation = params.get("orientation")
    if orientation is None:
        horizontal = "y" in given and "x" not in given and "y" not in given.after_stat()
        target = "x" if horizontal else "y"
    else:
        target = "x" if orientation == "y" else "y"
    if target in given or target in {standardise_aes_name(k) for k in params}:
        return mapping
    return Aes({target: after_stat("density")}).merge(given)


@register
def stat_ydensity(
    mapping=None,
    data=None,
    *,
    geom="violin",
    position="dodge",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Violin densities of ``y`` per x drawn with ``geom`` (default
    ``geom_violin``); see :mod:`ggmanim.stats.density`. Returns a layer."""

    return _stat_ctor("ydensity", geom, mapping, data, position, inherit_aes, name, params)
