"""stat_smooth: smoothed conditional means with a confidence band.

Methods (``method=``):

* ``"lm"`` -- polynomial least squares, ``y ~ poly(x, degree)`` (``degree=1``).
* ``"loess"`` -- local polynomial regression as R's ``loess`` (``span=0.75``,
  ``degree=2``; see :mod:`ggmanim.stats._loess`), bands from t quantiles with the
  loess "lookup" degrees of freedom, like ggplot2.
* ``"auto"`` -- ggplot2's rule: ``"loess"`` when the largest group has fewer than
  1000 rows, else ``"lm"`` (where ggplot2 switches to a GAM), with ggplot2's
  message (a ``UserWarning``) saying which method was used.
* a callable ``method(x, y, **method_args)`` returning a predictor: a callable
  (or an object with ``.predict``) taking the x grid and returning the fitted
  values, or a pair ``(fit, se)``; with ``se`` the band is ``fit +- z * se``
  with the normal quantile ``z``. This wraps any model (scikit-learn,
  statsmodels, splines ...).

``method_args`` (ggplot2's ``method.args``) are passed to a callable method; for
``"loess"`` they may set ``span``, ``degree`` and ``surface``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from statistics import NormalDist

import numpy as np
import pandas as pd

from ..registry import register
from ._loess import loess
from .base import Stat, scale_limits

try:  # optional dependency: exact t quantiles
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    _scipy_stats = None

__all__ = ["StatSmooth", "stat_smooth", "t_quantile"]

_METHODS = ("lm", "loess", "auto")

#: ggplot2's threshold for ``method="auto"``: loess below, a global fit at or above.
AUTO_LOESS_MAX = 1000


def t_quantile(p: float, dof: float) -> float:
    """Quantile ``p`` of Student's t with ``dof`` degrees of freedom.

    Uses ``scipy.stats.t`` when scipy is importable; otherwise falls back to the
    standard normal quantile (exact as ``dof -> inf``, slightly narrow bands for
    small samples).
    """
    if _scipy_stats is not None:
        return float(_scipy_stats.t.ppf(p, dof))
    return NormalDist().inv_cdf(p)


def _check_method(method) -> None:
    if not callable(method) and method not in _METHODS:
        raise NotImplementedError(
            f"stat_smooth method={method!r} is not supported "
            "(available: 'lm', 'loess', 'auto' or a callable)"
        )


def _is(method, name: str) -> bool:
    return isinstance(method, str) and method == name


class StatSmooth(Stat):
    """Fits ``y ~ x`` per group (see the module docstring for ``method``).

    Output columns: ``x``, ``y`` and, when ``se=True`` and the method gives
    standard errors, ``ymin``, ``ymax``, ``se``. The grid is ``n`` points over the
    group's x range, or over the x scale's limits with ``fullrange=True``. Groups
    with fewer than ``degree + 1`` distinct x values (at least 2) are dropped; so
    are groups a loess or callable fit fails on, with a warning.
    """

    required_aes = ("x", "y")
    default_aes = {}

    def __init__(
        self,
        method="lm",
        degree=None,
        span=0.75,
        n=80,
        se=True,
        level=0.95,
        fullrange=False,
        method_args=None,
        **params,
    ):
        super().__init__(
            method=method,
            degree=degree,
            span=span,
            n=n,
            se=se,
            level=level,
            fullrange=fullrange,
            method_args=method_args,
            **params,
        )

    def compute_layer(self, df: pd.DataFrame, scales=None, **params) -> pd.DataFrame:
        """Resolves ``method="auto"`` from the largest group, then fits per group."""
        method = params.get("method", "lm")
        _check_method(method)
        if _is(method, "auto") and len(df):
            keys = [c for c in ("PANEL", "group") if c in df.columns]
            largest = int(df.groupby(keys).size().max()) if keys else len(df)
            method = "loess" if largest < AUTO_LOESS_MAX else "lm"
            warnings.warn(
                f"`geom_smooth()` using method = '{method}' and formula = 'y ~ x'", stacklevel=2
            )
            params = {**params, "method": method}
        return super().compute_layer(df, scales, **params)

    def compute_group(  # type: ignore[override]
        self,
        df: pd.DataFrame,
        scales=None,
        method="lm",
        degree=None,
        span=0.75,
        n=80,
        se=True,
        level=0.95,
        fullrange=False,
        method_args=None,
    ) -> pd.DataFrame:
        _check_method(method)
        if _is(method, "auto"):  # compute_group called directly
            method = "loess" if len(df) < AUTO_LOESS_MAX else "lm"
        method_args = dict(method_args or {})
        if _is(method, "loess"):
            span = method_args.pop("span", span)
            degree = method_args.pop("degree", degree)
        if degree is None:
            degree = 2 if _is(method, "loess") else 1
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

        rng = scale_limits(scales, "x") if fullrange else None
        if rng is None:
            rng = (x.min(), x.max())
        xseq = np.linspace(rng[0], rng[1], int(n))

        if callable(method):
            fit = _predict_callable(method, x, y, xseq, se, level, method_args)
        elif _is(method, "loess"):
            fit = _predict_loess(x, y, xseq, se, level, span, degree, method_args)
        else:
            fit = _predict_lm(x, y, xseq, se, level, degree)
        if fit is None:
            return pd.DataFrame()
        return pd.DataFrame({"x": xseq, **fit})


def _band(yhat, se_fit, q) -> dict[str, np.ndarray]:
    return {"y": yhat, "ymin": yhat - q * se_fit, "ymax": yhat + q * se_fit, "se": se_fit}


def _predict_lm(x, y, xseq, se, level, degree) -> dict[str, np.ndarray]:
    # Centre/scale x for a well-conditioned Vandermonde matrix.
    mu, sd = x.mean(), x.std() or 1.0
    X = np.vander((x - mu) / sd, degree + 1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    Xs = np.vander((xseq - mu) / sd, degree + 1)
    yhat = Xs @ beta
    if not se:
        return {"y": yhat}
    dof = len(x) - (degree + 1)
    if dof <= 0:
        return _band(yhat, np.full_like(xseq, np.nan), np.nan)
    resid = y - X @ beta
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se_fit = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Xs, cov, Xs), 0.0))
    return _band(yhat, se_fit, t_quantile((1 + level) / 2, dof))


def _predict_loess(x, y, xseq, se, level, span, degree, method_args):
    try:
        model = loess(x, y, span=float(span), degree=degree, **method_args)
    except (ValueError, np.linalg.LinAlgError) as e:
        warnings.warn(f"stat_smooth: loess failed for a group ({e}); dropped", stacklevel=4)
        return None
    if not se:
        return {"y": model.predict(xseq)}
    yhat, se_fit = model.predict(xseq, se=True)
    return _band(yhat, se_fit, t_quantile((1 + level) / 2, model.df))


def _predict_callable(method: Callable, x, y, xseq, se, level, method_args):
    try:
        model = method(x, y, **method_args)
        predict = model if callable(model) else model.predict
        res = predict(xseq)
    except Exception as e:
        name = getattr(method, "__name__", type(method).__name__)
        warnings.warn(f"stat_smooth: method {name} failed for a group ({e}); dropped", stacklevel=4)
        return None
    if isinstance(res, tuple):
        yhat, se_fit = (np.asarray(v, dtype=float).reshape(-1) for v in res)
    else:
        yhat, se_fit = np.asarray(res, dtype=float).reshape(-1), None
    if len(yhat) != len(xseq):
        raise ValueError(
            f"stat_smooth: the callable method returned {len(yhat)} values for {len(xseq)} x"
        )
    if not se or se_fit is None:
        return {"y": yhat}
    return _band(yhat, se_fit, NormalDist().inv_cdf((1 + level) / 2))


@register
def stat_smooth(
    method="lm",
    degree=None,
    span=0.75,
    n=80,
    se=True,
    level=0.95,
    fullrange=False,
    method_args=None,
    **params,
) -> StatSmooth:
    """Smoothed conditional mean: ``method`` is ``"lm"`` (polynomial of ``degree``,
    default 1), ``"loess"`` (``span``; ``degree`` default 2), ``"auto"`` or a
    callable (see :mod:`ggmanim.stats.smooth`).

    ``fullrange=True`` evaluates over the x scale's limits
    (``scales["x"].get_limits()``) instead of the group's data range.
    """
    return StatSmooth(
        method=method,
        degree=degree,
        span=span,
        n=n,
        se=se,
        level=level,
        fullrange=fullrange,
        method_args=method_args,
        **params,
    )
