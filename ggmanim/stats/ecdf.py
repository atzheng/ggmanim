"""stat_ecdf: empirical cumulative distribution function."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import register
from .base import Stat

__all__ = ["StatEcdf", "stat_ecdf"]


class StatEcdf(Stat):
    """The ECDF of ``x`` per group.

    Output columns: ``x`` (the sorted distinct values, or ``n`` evenly spaced
    points over the group's range when ``n`` is given), ``ecdf`` and ``y`` (the
    fraction of observations ``<= x``, weighted by the ``weight`` column when
    present). With ``pad=True`` (default) a row ``(-inf, 0)`` is prepended and
    ``(inf, 1)`` appended, so a step geom runs from the panel's left edge at 0 to
    its right edge at 1 (the geom resolves the infinities to the panel range).
    """

    required_aes = ("x",)
    default_aes = {}
    dropped_aes = ("weight", "y")

    def __init__(self, n=None, pad=True, **params):
        super().__init__(n=n, pad=pad, **params)

    def compute_group(  # type: ignore[override]
        self, df: pd.DataFrame, scales=None, n=None, pad=True
    ) -> pd.DataFrame:
        x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
        w = (
            pd.to_numeric(df["weight"], errors="coerce").to_numpy(dtype=float)
            if "weight" in df.columns
            else np.ones(len(x))
        )
        ok = np.isfinite(x) & np.isfinite(w)
        x, w = x[ok], w[ok]
        if len(x) == 0:
            return pd.DataFrame()
        order = np.argsort(x, kind="stable")
        xs, cw = x[order], np.cumsum(w[order])
        total = cw[-1]
        if n is None:
            xgrid = np.unique(xs)
        else:
            xgrid = np.linspace(xs[0], xs[-1], int(n))
        # fraction of the weight at or below each grid value
        idx = np.searchsorted(xs, xgrid, side="right")
        y = np.where(idx > 0, cw[np.maximum(idx - 1, 0)], 0.0) / total
        if pad:
            xgrid = np.concatenate([[-np.inf], xgrid, [np.inf]])
            y = np.concatenate([[0.0], y, [1.0]])
        return pd.DataFrame({"x": xgrid, "ecdf": y, "y": y})


@register
def stat_ecdf(
    mapping=None,
    data=None,
    *,
    geom="step",
    position="identity",
    n=None,
    pad: bool = True,
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Layer drawing the empirical CDF of ``x`` (ggplot2's ``stat_ecdf``).

    ``geom`` defaults to a step function (``"step"``); ``n`` evaluates the ECDF
    at ``n`` evenly spaced points instead of every distinct value; ``pad=False``
    drops the 0 / 1 tails at the panel edges. Returns a layer.
    """
    from ..geoms.count import stat_layer  # geoms import manim: keep stats free of it

    params.update(n=n, pad=pad)
    return stat_layer(
        "ecdf",
        geom,
        mapping,
        data,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
