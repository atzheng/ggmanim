"""geom_step: stairstep lines (a :class:`~ggmanim.geoms.line.GeomPath` through
the step-expanded data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from manim import VMobject

from ..registry import register
from ..theme import Theme
from .base import finite_rows, make_layer
from .line import GeomPath

__all__ = ["GeomStep", "geom_step", "stairstep"]

_DIRECTIONS = ("hv", "vh", "mid")


def stairstep(data: pd.DataFrame, direction: str = "hv") -> pd.DataFrame:
    """One group's rows (sorted by ``x``) expanded into the vertices of a
    stairstep, like ggplot2's ``stairstep()``.

    ``"hv"`` goes horizontally first, then vertically (``2n - 1`` vertices);
    ``"vh"`` vertically first (``2n - 1``); ``"mid"`` changes level half-way
    between adjacent x values (``2n`` vertices). The other columns of a vertex
    come from the row whose ``x`` it uses (``"mid"``: whose ``y``), as in ggplot2.
    """
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}, got {direction!r}")
    data = data.sort_values("x", kind="stable").reset_index(drop=True)
    n = len(data)
    if n <= 1:
        return data
    x = data["x"].to_numpy(dtype=float)
    y = data["y"].to_numpy(dtype=float)
    if direction == "mid":
        mids = (x[:-1] + x[1:]) / 2
        xs = np.concatenate([[x[0]], np.repeat(mids, 2), [x[-1]]])
        rows = np.repeat(np.arange(n), 2)
        ys = y[rows]
        src = rows
    else:
        xi = np.repeat(np.arange(n), 2)[1:]  # 0 1 1 2 2 ...
        yi = np.repeat(np.arange(n), 2)[:-1]  # 0 0 1 1 2 ...
        if direction == "vh":
            xi, yi = yi, xi
        xs, ys = x[xi], y[yi]
        src = xi
    out = data.iloc[src].reset_index(drop=True)
    out["x"] = xs
    out["y"] = ys
    return out


class GeomStep(GeomPath):
    """Stairsteps: each group is sorted by ``x``, expanded by :func:`stairstep`
    (``direction="hv"``, ``"vh"`` or ``"mid"``) and drawn as a path, so marks,
    labels, clipping, dashing and per-segment colours work as for
    :class:`GeomPath` (one mark per group)."""

    extra_params: tuple[str, ...] = ("label", "direction")

    def draw_panel(
        self,
        data,
        panel_params,
        coord,
        theme: Theme | None = None,
        label=None,
        direction: str = "hv",
        **params,
    ) -> VMobject:
        data = finite_rows(data, ("x", "y"))
        if "group" not in data.columns:
            data = self.use_defaults(data, {})
        pieces = [stairstep(g, direction) for _, g in data.groupby("group", sort=True)]
        if pieces:
            data = pd.concat(pieces, ignore_index=True)
        return super().draw_panel(data, panel_params, coord, theme, label=label, **params)


@register
def geom_step(
    mapping=None,
    data=None,
    *,
    direction: str = "hv",
    stat="identity",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Stairstep lines connecting observations in order of x, one per group.

    ``direction``: ``"hv"`` (horizontal then vertical, the default), ``"vh"``
    or ``"mid"`` (steps half-way between x values). ``label=`` as for
    :func:`~ggmanim.geoms.line.geom_line`.
    """
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}, got {direction!r}")
    params["direction"] = direction
    return make_layer(
        GeomStep(),
        mapping,
        data,
        stat=stat,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
