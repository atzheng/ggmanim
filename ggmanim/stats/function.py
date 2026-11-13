"""stat_function: evaluates a function on a grid of x values."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import NO_GROUP, Stat, scale_limits

try:
    from ..registry import register
except ImportError:  # registry not available yet

    def register(f):
        return f


__all__ = ["StatFunction", "stat_function", "evaluate_function"]


def evaluate_function(fun, x: np.ndarray, args=None) -> np.ndarray:
    """Evaluate ``fun`` at ``x``, vectorised if possible.

    ``fun(x)`` is tried first with the whole array; if it raises or does not
    return an array of the same shape it is applied element-wise via
    ``np.vectorize``.
    """
    args = dict(args or {})
    try:
        with np.errstate(all="ignore"):
            y = np.asarray(fun(x, **args), dtype=float)
        if y.shape == x.shape:
            return y
    except Exception:
        pass
    with np.errstate(all="ignore"):
        return np.vectorize(lambda v: fun(v, **args), otypes=[float])(x)


def _transformation(scale):
    """Duck-typed access to a continuous scale's transformation object, if any."""
    if scale is None:
        return None
    if hasattr(scale, "get_transformation"):
        return scale.get_transformation()
    return getattr(scale, "trans", None)


class StatFunction(Stat):
    """Ignores the layer data and evaluates ``fun`` on ``n`` points.

    The grid spans ``xlim`` if given, else the x scale's limits
    (``scales["x"].get_limits()``), else the range of the data's ``x`` column.
    Like ggplot2, the grid is in *transformed* x space: if the x scale has a
    transformation (``scale.get_transformation()`` or ``scale.trans`` with
    ``inverse``/``transform``), ``fun`` is called with back-transformed x and
    its result is transformed with the y scale's transformation.
    """

    required_aes = ()
    default_aes = {}

    def __init__(self, fun=None, n=101, xlim=None, args=None, **params):
        if fun is None or not callable(fun):
            raise TypeError("stat_function requires a callable `fun`")
        super().__init__(fun=fun, n=n, xlim=xlim, args=args, **params)

    def compute_layer(self, df: pd.DataFrame, scales=None, **params) -> pd.DataFrame:
        params = self._filter_params(params)
        if df is None:
            df = pd.DataFrame()
        if "PANEL" not in df.columns:
            return self.compute_group(df, scales, **params).assign(group=NO_GROUP)
        pieces = [
            self.compute_group(pdf, scales, **params).assign(PANEL=panel, group=NO_GROUP)
            for panel, pdf in df.groupby("PANEL", sort=True, observed=True)
        ]
        return pd.concat(pieces, ignore_index=True)

    def compute_panel(self, df: pd.DataFrame, scales=None, **params) -> pd.DataFrame:
        return self.compute_group(df, scales, **params).assign(group=NO_GROUP)

    def compute_group(self, df: pd.DataFrame, scales=None, fun=None, n=101,
                      xlim=None, args=None) -> pd.DataFrame:
        if fun is None:
            raise TypeError("stat_function requires a callable `fun`")
        rng = tuple(float(v) for v in xlim) if xlim is not None else None
        if rng is None:
            rng = scale_limits(scales, "x")
        if rng is None and df is not None and "x" in df.columns:
            x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if len(x):
                rng = (x.min(), x.max())
        if rng is None:
            raise ValueError(
                "stat_function needs an x range: pass xlim=(lo, hi), set x scale "
                "limits, or add a layer with x data"
            )

        xseq = np.linspace(rng[0], rng[1], int(n))
        xs = (scales or {}).get("x")
        xt = _transformation(xs)
        x_data = xt.inverse(xseq) if xt is not None and hasattr(xt, "inverse") else xseq
        y = evaluate_function(fun, np.asarray(x_data, dtype=float), args)
        yt = _transformation((scales or {}).get("y"))
        if yt is not None and hasattr(yt, "transform"):
            with np.errstate(all="ignore"):
                y = np.asarray(yt.transform(y), dtype=float)
        return pd.DataFrame({"x": xseq, "y": y})


@register
def stat_function(fun, n=101, xlim=None, args=None, **params) -> StatFunction:
    """Compute ``y = fun(x, **args)`` on ``n`` evenly spaced x values."""
    return StatFunction(fun=fun, n=n, xlim=xlim, args=args, **params)
