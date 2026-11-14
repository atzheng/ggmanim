"""stat_qq / stat_qq_line: quantile-quantile points and reference line.

Theoretical quantiles default to the standard normal (:func:`qnorm`: Acklam's
rational approximation refined by one Halley step on ``math.erfc``, accurate to
about 1e-15). ``distribution`` may be any quantile function ``q(p, **dparams)``,
e.g. ``scipy.stats.t.ppf`` with ``dparams={"df": 5}``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..registry import register
from .base import Stat, scale_limits

__all__ = ["StatQq", "StatQqLine", "qnorm", "ppoints", "stat_qq", "stat_qq_line"]

# Acklam's coefficients (https://web.archive.org/web/2007/home.online.no/~pjacklam/notes/invnorm/)
_A = (-39.69683028665376, 220.9460984245205, -275.9285104469687,
      138.3577518672690, -30.66479806614716, 2.506628277459239)  # fmt: skip
_B = (-54.47609879822406, 161.5858368580409, -155.6989798598866,
      66.80131188771972, -13.28068155288572)  # fmt: skip
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838,
      -2.549732539343734, 4.374664141464968, 2.938163982698783)  # fmt: skip
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996, 3.754408661907416)
_P_LOW = 0.02425


def _poly(c, x):
    out = 0.0
    for v in c:
        out = out * x + v
    return out


def _qnorm1(p: float) -> float:
    if not 0.0 <= p <= 1.0 or math.isnan(p):
        return math.nan
    if p == 0.0:
        return -math.inf
    if p == 1.0:
        return math.inf
    if p > 0.5:  # by symmetry: 1 - p is exact here, and Phi(x) - p loses no digits
        return -_qnorm1(1.0 - p)
    if p < _P_LOW:
        q = math.sqrt(-2 * math.log(p))
        x = _poly(_C, q) / (_poly(_D, q) * q + 1)
    else:
        q = p - 0.5
        r = q * q
        x = _poly(_A, r) * q / (_poly(_B, r) * r + 1)
    # one Halley step on Phi(x) - p, Phi via erfc
    e = 0.5 * math.erfc(-x / math.sqrt(2)) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def qnorm(p, mean: float = 0.0, sd: float = 1.0):
    """Normal quantile function (vectorised; ``0 -> -inf``, ``1 -> inf``)."""
    arr = np.asarray(p, dtype=float)
    z = np.vectorize(_qnorm1, otypes=[float])(arr) if arr.size else arr.copy()
    out = mean + sd * z
    return float(out) if np.ndim(out) == 0 else out


def ppoints(n: int) -> np.ndarray:
    """R's ``ppoints(n)``: ``(i - a) / (n + 1 - 2a)``, ``a = 3/8`` if ``n <= 10`` else 1/2."""
    a = 3 / 8 if n <= 10 else 0.5
    return (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)


def _theoretical(p, distribution, dparams) -> np.ndarray:
    fun = qnorm if distribution is None else distribution
    return np.asarray(fun(p, **dict(dparams or {})), dtype=float)


def _sample(df: pd.DataFrame) -> np.ndarray:
    s = pd.to_numeric(df["sample"], errors="coerce").to_numpy(dtype=float)
    return np.sort(s[np.isfinite(s)])


class StatQq(Stat):
    """Sample quantiles against theoretical quantiles, per group.

    Output columns: ``sample`` and ``y`` (the sorted sample), ``theoretical`` and
    ``x`` (``distribution(ppoints(n), **dparams)``).
    """

    required_aes = ("sample",)
    default_aes = {}
    dropped_aes = ("sample", "x", "y")

    def __init__(self, distribution=None, dparams=None, **params):
        super().__init__(distribution=distribution, dparams=dparams, **params)

    def compute_group(  # type: ignore[override]
        self, df: pd.DataFrame, scales=None, distribution=None, dparams=None
    ) -> pd.DataFrame:
        sample = _sample(df)
        if len(sample) == 0:
            return pd.DataFrame()
        theoretical = _theoretical(ppoints(len(sample)), distribution, dparams)
        return pd.DataFrame(
            {"x": theoretical, "y": sample, "sample": sample, "theoretical": theoretical}
        )


class StatQqLine(Stat):
    """The line through the ``line_p`` (default 25% / 75%) quantile pairs, per group.

    Output: two rows (``x`` over the theoretical quantiles' range, or the x scale's
    limits with ``fullrange=True``) with ``y`` on the line, plus ``slope`` and
    ``intercept`` columns. Sample quantiles use R's default (type 7) definition.
    """

    required_aes = ("sample",)
    default_aes = {}
    dropped_aes = ("sample", "x", "y")

    def __init__(
        self, distribution=None, dparams=None, line_p=(0.25, 0.75), fullrange=False, **params
    ):
        super().__init__(
            distribution=distribution,
            dparams=dparams,
            line_p=line_p,
            fullrange=fullrange,
            **params,
        )

    def compute_group(  # type: ignore[override]
        self,
        df: pd.DataFrame,
        scales=None,
        distribution=None,
        dparams=None,
        line_p=(0.25, 0.75),
        fullrange=False,
    ) -> pd.DataFrame:
        sample = _sample(df)
        if len(sample) < 2:
            return pd.DataFrame()
        p = np.asarray(line_p, dtype=float)
        if p.shape != (2,) or not (0 < p[0] < p[1] < 1):
            raise ValueError(f"line_p must be two increasing probabilities, got {line_p!r}")
        yq = np.quantile(sample, p)
        xq = _theoretical(p, distribution, dparams)
        slope = float((yq[1] - yq[0]) / (xq[1] - xq[0]))
        intercept = float(yq[0] - slope * xq[0])
        rng = scale_limits(scales, "x") if fullrange else None
        if rng is None:
            theoretical = _theoretical(ppoints(len(sample)), distribution, dparams)
            rng = (float(np.min(theoretical)), float(np.max(theoretical)))
        x = np.asarray(rng, dtype=float)
        return pd.DataFrame(
            {"x": x, "y": intercept + slope * x, "slope": slope, "intercept": intercept}
        )


@register
def stat_qq(
    mapping=None,
    data=None,
    *,
    geom="point",
    position="identity",
    distribution=None,
    dparams=None,
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Layer of sample vs theoretical quantiles of ``aes(sample=...)`` (as
    :func:`~ggmanim.geoms.qq.geom_qq`). Returns a layer."""
    from ..geoms.count import stat_layer  # geoms import manim: keep stats free of it

    params.update(distribution=distribution, dparams=dparams)
    return stat_layer(
        "qq",
        geom,
        mapping,
        data,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )


@register
def stat_qq_line(
    mapping=None,
    data=None,
    *,
    geom="path",
    position="identity",
    distribution=None,
    dparams=None,
    line_p=(0.25, 0.75),
    fullrange: bool = False,
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Layer of the Q-Q reference line through the ``line_p`` quantiles (as
    :func:`~ggmanim.geoms.qq.geom_qq_line`). Returns a layer."""
    from ..geoms.count import stat_layer  # geoms import manim: keep stats free of it

    params.update(distribution=distribution, dparams=dparams, line_p=line_p, fullrange=fullrange)
    return stat_layer(
        "qq_line",
        geom,
        mapping,
        data,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
