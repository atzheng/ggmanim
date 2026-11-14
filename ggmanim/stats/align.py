"""stat_align: interpolate every group onto a shared x grid (``geom_area``'s default
stat, as in ggplot2 >= 3.4).

Stacking adds up ``y`` per x position, so areas whose groups are observed at
different x values stack incorrectly unless each group has a value at every x.
:class:`StatAlign` builds one grid for the whole layer -- every x value of every
group, plus the points where a group's ``y`` crosses 0 -- and linearly interpolates
each group onto the part of the grid inside its own x range.

Like ggplot2, each grid point ``u`` is accompanied by ``u - adjust`` and
``u + adjust`` (``adjust`` is 0.1% of the x range, at most a third of the smallest
gap), so a group that starts or ends at ``u`` drops to 0 right next to it instead of
sloping down to the neighbouring x. Every group therefore gets one extra row at
each end (``y = 0``, ``align_padding = True``); geoms leave those rows out of their
outlines. Several rows at the same x (a vertical jump) keep their first and last
``y``, the first moved ``adjust`` to the left.

``orientation="y"`` swaps the roles of the axes: groups are interpolated onto a
shared *y* grid, giving ``x`` per ``y`` (output marked ``flipped_aes = True``).
With ``orientation=None`` (the default) the stat is flipped when the y scale is
discrete and the x scale is not (ggplot2's ``has_flipped_aes``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import register
from .base import Stat
from .summary import _check_orientation, _flipped

__all__ = ["StatAlign", "stat_align"]


def _xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    return x, y


def _zero_crossings(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """x where the path through ``(x, y)`` (sorted by x) changes sign (``y < 0``)."""
    neg = y < 0
    i = np.flatnonzero(neg[:-1] != neg[1:])
    dy = y[i + 1] - y[i]
    with np.errstate(all="ignore"):
        cross = x[i] - y[i] * (x[i + 1] - x[i]) / dy
    return cross[np.isfinite(cross)]


def align_grid(df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """The shared grid and ``adjust`` for a layer's data (see the module docstring)."""
    locs = []
    groups = df.groupby("group", sort=True) if "group" in df.columns else [(None, df)]
    for _, gdf in groups:
        x, y = _xy(gdf)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]
        locs.append(x)
        if len(x) > 1:
            locs.append(_zero_crossings(x, y))
    loc = np.unique(np.concatenate(locs)) if locs else np.zeros(0)
    if len(loc) < 2:
        return loc, 0.0
    adjust = min((loc[-1] - loc[0]) * 0.001, float(np.min(np.diff(loc))) / 3)
    return np.unique(np.concatenate([loc - adjust, loc, loc + adjust])), float(adjust)


class StatAlign(Stat):
    """Interpolate each group onto the layer's shared x grid (see the module
    docstring). Output columns: ``x``, ``y``, ``align_padding`` (``True`` on the
    two zero-height end rows) plus the group's constant columns; with
    ``orientation="y"`` x and y swap roles and ``flipped_aes`` is ``True``."""

    required_aes = ("x", "y")

    def __init__(self, orientation=None, **params):
        _check_orientation(orientation)
        super().__init__(orientation=orientation, **params)

    def compute_layer(self, df: pd.DataFrame, scales=None, **params) -> pd.DataFrame:
        self.check_required_aes(df)
        params = dict(params)
        if _flipped(params.pop("orientation", None), scales):
            swap = {"x": "y", "y": "x"}
            res = self.compute_layer(df.rename(columns=swap), scales, **params)
            return res.rename(columns=swap).assign(flipped_aes=True) if len(res) else res
        params["unique_loc"], params["adjust"] = align_grid(df)
        return super().compute_layer(df, scales, **params)

    def compute_group(  # type: ignore[override]
        self, df: pd.DataFrame, scales=None, unique_loc=None, adjust: float = 0.0
    ) -> pd.DataFrame:
        x, y = _xy(df)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(np.unique(x)) < 2:  # not enough data to align
            return pd.DataFrame()
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]
        # several rows at one x: keep the first (moved left) and the last
        keep = np.ones(len(x), dtype=bool)
        first = np.r_[True, x[1:] != x[:-1]]
        last = np.r_[x[1:] != x[:-1], True]
        keep &= first | last
        x = np.where(first & ~last, x - adjust, x)
        x, y = x[keep], y[keep]
        if unique_loc is None:
            unique_loc, adjust = align_grid(df)
        loc = np.asarray(unique_loc, dtype=float)
        loc = loc[(loc >= x[0]) & (loc <= x[-1])]
        yv = np.interp(loc, x, y)
        xs = np.concatenate([[loc[0] - adjust], loc, [loc[-1] + adjust]])
        ys = np.concatenate([[0.0], yv, [0.0]])
        pad = np.zeros(len(xs), dtype=bool)
        pad[[0, -1]] = True
        return pd.DataFrame({"x": xs, "y": ys, "align_padding": pad})


@register
def stat_align(
    mapping=None,
    data=None,
    *,
    geom="area",
    position="stack",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Layer of groups interpolated onto a shared x grid (default: stacked areas).
    Also usable as ``geom_*(stat="align")``; ``geom_area``'s default. Returns a
    layer. ``orientation="y"`` aligns ``x`` on a shared y grid (see the module
    docstring)."""
    from ..geoms.count import stat_layer  # geoms import manim: keep stats free of it

    return stat_layer(
        "align",
        geom,
        mapping,
        data,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
