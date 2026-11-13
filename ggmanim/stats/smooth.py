"""stat_smooth: polynomial least-squares fit with a confidence band."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from ..registry import register
from .base import Stat, scale_limits

try:  # optional dependency: exact t quantiles
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    _scipy_stats = None

__all__ = ["StatSmooth", "stat_smooth", "t_quantile"]

_METHODS = ("lm", "auto")


def t_quantile(p: float, dof: float) -> float:
    """Quantile ``p`` of Student's t with ``dof`` degrees of freedom.

    Uses ``scipy.stats.t`` when scipy is importable; otherwise falls back to the
    standard normal quantile (exact as ``dof -> inf``, slightly narrow bands for
    small samples).
    """
    if _scipy_stats is not None:
        return float(_scipy_stats.t.ppf(p, dof))
    return NormalDist().inv_cdf(p)


class StatSmooth(Stat):
    """Fits ``y ~ poly(x, degree)`` per group by least squares.

    Output columns: ``x``, ``y`` and, when ``se=True``, ``ymin``, ``ymax``, ``se``.
    Groups with fewer than ``degree + 1`` distinct x values are dropped.
    """

    required_aes = ("x", "y")
    default_aes = {}

    def __init__(self, method="lm", degree=1, n=80, se=True, level=0.95, fullrange=False, **params):
        super().__init__(
            method=method, degree=degree, n=n, se=se, level=level, fullrange=fullrange, **params
        )

    def compute_group(  # type: ignore[override]
        self,
        df: pd.DataFrame,
        scales=None,
        method="lm",
        degree=1,
        n=80,
        se=True,
        level=0.95,
        fullrange=False,
    ) -> pd.DataFrame:
        if method not in _METHODS:
            raise NotImplementedError(
                f"stat_smooth method={method!r} is not supported (available: 'lm')"
            )
        degree = int(degree)
        if degree < 0:
            raise ValueError("degree must be >= 0")
        if not 0 < level < 1:
            raise ValueError("level must be in (0, 1)")

        x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(np.unique(x)) < max(degree + 1, 2):
            return pd.DataFrame()

        # Centre/scale x for a well-conditioned Vandermonde matrix.
        mu, sd = x.mean(), x.std() or 1.0
        X = np.vander((x - mu) / sd, degree + 1)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)

        rng = scale_limits(scales, "x") if fullrange else None
        if rng is None:
            rng = (x.min(), x.max())
        xseq = np.linspace(rng[0], rng[1], int(n))
        Xs = np.vander((xseq - mu) / sd, degree + 1)
        yhat = Xs @ beta
        out = {"x": xseq, "y": yhat}

        if se:
            dof = len(x) - (degree + 1)
            if dof > 0:
                resid = y - X @ beta
                sigma2 = float(resid @ resid) / dof
                cov = sigma2 * np.linalg.pinv(X.T @ X)
                se_fit = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Xs, cov, Xs), 0.0))
                q = t_quantile((1 + level) / 2, dof)
            else:
                se_fit = np.full_like(xseq, np.nan)
                q = np.nan
            out["ymin"] = yhat - q * se_fit
            out["ymax"] = yhat + q * se_fit
            out["se"] = se_fit
        return pd.DataFrame(out)


@register
def stat_smooth(
    method="lm", degree=1, n=80, se=True, level=0.95, fullrange=False, **params
) -> StatSmooth:
    """Smoothed conditional mean (``lm``: polynomial of ``degree`` in x).

    ``fullrange=True`` evaluates over the x scale's limits
    (``scales["x"].get_limits()``) instead of the group's data range.
    """
    return StatSmooth(
        method=method, degree=degree, n=n, se=se, level=level, fullrange=fullrange, **params
    )
