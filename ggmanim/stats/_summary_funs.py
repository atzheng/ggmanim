"""Summary functions for ``stat_summary(fun_data=...)`` (ggplot2 / Hmisc names).

Each takes a 1-d array of finite values (plus keyword options) and returns a
dict with ``y``, ``ymin`` and ``ymax`` (NaN where undefined, e.g. the standard
error of a single value). Pure numpy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .smooth import t_quantile

__all__ = ["mean_se", "mean_cl_normal", "mean_sdl", "median_hilow", "SUMMARY_FUNS"]


def _values(x: Any) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def _sd(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1)) if len(x) > 1 else np.nan


def mean_se(x: Any, mult: float = 1.0) -> dict[str, float]:
    """Mean +/- ``mult`` standard errors (``sd / sqrt(n)``)."""
    x = _values(x)
    if len(x) == 0:
        return {"y": np.nan, "ymin": np.nan, "ymax": np.nan}
    m = float(np.mean(x))
    se = _sd(x) / np.sqrt(len(x))
    return {"y": m, "ymin": m - mult * se, "ymax": m + mult * se}


def mean_cl_normal(x: Any, conf_int: float = 0.95, mult: float | None = None) -> dict[str, float]:
    """Mean with a t-based confidence interval (``Hmisc::smean.cl.normal``):
    ``mean +/- t_{(1 + conf_int)/2, n-1} * sd / sqrt(n)`` (``mult`` overrides the
    t quantile). Uses the normal quantile if scipy is unavailable."""
    x = _values(x)
    if len(x) == 0:
        return {"y": np.nan, "ymin": np.nan, "ymax": np.nan}
    m = float(np.mean(x))
    if len(x) < 2:
        return {"y": m, "ymin": np.nan, "ymax": np.nan}
    q = float(mult) if mult is not None else t_quantile((1 + conf_int) / 2, len(x) - 1)
    half = q * _sd(x) / np.sqrt(len(x))
    return {"y": m, "ymin": m - half, "ymax": m + half}


def mean_sdl(x: Any, mult: float = 2.0) -> dict[str, float]:
    """Mean +/- ``mult`` standard deviations (``Hmisc::smean.sdl``)."""
    x = _values(x)
    if len(x) == 0:
        return {"y": np.nan, "ymin": np.nan, "ymax": np.nan}
    m = float(np.mean(x))
    sd = _sd(x)
    return {"y": m, "ymin": m - mult * sd, "ymax": m + mult * sd}


def median_hilow(x: Any, conf_int: float = 0.95) -> dict[str, float]:
    """Median and the outer quantiles ``(1 -/+ conf_int) / 2``
    (``Hmisc::smedian.hilow``; linear-interpolation quantiles, R's type 7)."""
    x = _values(x)
    if len(x) == 0:
        return {"y": np.nan, "ymin": np.nan, "ymax": np.nan}
    lo, med, hi = np.quantile(x, [(1 - conf_int) / 2, 0.5, (1 + conf_int) / 2])
    return {"y": float(med), "ymin": float(lo), "ymax": float(hi)}


#: ``fun_data`` names accepted by :func:`~ggmanim.stats.summary.stat_summary`.
SUMMARY_FUNS: dict[str, Callable[..., dict[str, float]]] = {
    "mean_se": mean_se,
    "mean_cl_normal": mean_cl_normal,
    "mean_sdl": mean_sdl,
    "median_hilow": median_hilow,
}
