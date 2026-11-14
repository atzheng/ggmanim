"""geom_rug: short ticks along the panel edges marking each observation.

For every row, ``x`` values get a tick on the bottom (``"b"``) and/or top
(``"t"``) edge and ``y`` values on the left (``"l"``) and/or right (``"r"``) edge,
as chosen by ``sides`` (default ``"bl"``). ``length`` is the tick length in scene
units (default 3% of the panel's height for x ticks / width for y ticks, as
ggplot2's ``unit(0.03, "npc")``); ``outside=True`` draws the ticks outward from
the edges instead of into the panel.

Edges are the panel's axis ranges (``"b"``/``"l"`` = the start of the y / x
range), so under ``coord_flip`` the x ticks sit on the left edge, as in ggplot2.
Values outside the panel are dropped when clipping is on. One mark per row with
at least one visible tick: a ``VMobject`` with one subpath per tick.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from manim import DOWN, UP, Line, VGroup, VMobject

from ..registry import register
from ..theme import Theme, theme_default
from ._clip import panel_rect
from ._linetype import apply_linetype
from .base import (
    KEY_SIZE,
    Geom,
    attach_marks,
    make_layer,
    multi_path_mobject,
    num,
    paint_stroke,
    resolve_color,
    transform_points,
)

__all__ = ["GeomRug", "geom_rug"]


def _check_sides(sides: Any) -> str:
    sides = str(sides)
    if any(c not in "tblr" for c in sides):
        raise ValueError(f"sides must be a string of 't', 'b', 'l', 'r', got {sides!r}")
    return sides


class GeomRug(Geom):
    """Rug ticks (see the module docstring)."""

    default_aes = {"color": None, "linewidth": 1.0, "alpha": 1.0, "linetype": "solid"}
    extra_params: tuple[str, ...] = ("sides", "length", "outside")

    def handle_na(
        self, data: pd.DataFrame, params: Mapping[str, Any] | None = None, label: str | None = None
    ) -> pd.DataFrame:
        """Rows are kept when either ``x`` or ``y`` is finite (each draws what it can)."""
        return data

    def draw_panel(
        self,
        data,
        panel_params,
        coord,
        theme: Theme | None = None,
        sides: str = "bl",
        length: float | None = None,
        outside: bool = False,
        **params,
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        sides = _check_sides(sides)
        if "x" not in data.columns and "y" not in data.columns:
            raise ValueError("GeomRug requires an x or y aesthetic")
        data = self.use_defaults(data, params)
        out, drawn = VGroup(), []
        if not len(data):
            return attach_marks(out, data, per="row")
        if panel_params is not None:
            xr, yr = panel_params.x_range, panel_params.y_range
        else:  # no panel: span the data
            xr = _span(data, "x")
            yr = _span(data, "y")
        clip = panel_rect(coord, panel_params) is not None
        fg = theme.get_foreground()
        edges = {"b": yr[0], "t": yr[1], "l": xr[0], "r": xr[1]}
        for i, row in enumerate(data.to_dict("records")):
            pieces = []
            for side in sides:
                along = "x" if side in "bt" else "y"
                v = num(row.get(along), np.nan)
                rng = xr if along == "x" else yr
                if not np.isfinite(v):
                    continue
                if clip and not (min(rng) - 1e-9 <= v <= max(rng) + 1e-9):
                    continue
                other = yr if along == "x" else xr
                at = edges[side]
                far = other[1] if at == other[0] else other[0]
                xs, ys = ([v, v], [at, far]) if along == "x" else ([at, far], [v, v])
                p0, p1 = transform_points(coord, xs, ys)
                d = p1 - p0
                span = float(np.linalg.norm(d))
                if span == 0:
                    continue
                size = 0.03 * span if length is None else float(length)
                step = d / span * size * (-1.0 if outside else 1.0)
                pieces.append(np.array([p0, p0 + step]))
            if pieces:
                color = resolve_color(row.get("color"), fg)
                mob = multi_path_mobject(
                    pieces, color, num(row.get("linewidth"), 1.0), num(row.get("alpha"), 1.0)
                )
                out.add(apply_linetype(mob, row.get("linetype")))
                drawn.append(i)
        return attach_marks(out, data.iloc[drawn], per="row")

    def draw_key(self, data_row, theme: Theme | None = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        row = self.key_aes(data_row)
        line = paint_stroke(
            Line(DOWN * KEY_SIZE / 2, UP * KEY_SIZE / 2),
            resolve_color(row.get("color"), theme.get_foreground()),
            num(row.get("linewidth"), 1.0),
            num(row.get("alpha"), 1.0),
        )
        return apply_linetype(line, row.get("linetype"), max_period=KEY_SIZE)


def _span(data: pd.DataFrame, col: str) -> tuple[float, float]:
    if col not in data.columns:
        return (0.0, 1.0)
    v = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return (float(v.min()), float(v.max())) if len(v) else (0.0, 1.0)


@register
def geom_rug(
    mapping=None,
    data=None,
    *,
    sides: str = "bl",
    length: float | None = None,
    outside: bool = False,
    stat="identity",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Rug plots: ticks on the panel edges at each ``x`` (bottom/top) and ``y``
    (left/right). ``sides`` is any of ``"tblr"`` (default ``"bl"``), ``length``
    the tick length in scene units (default 3% of the panel), ``outside`` draws
    them outward."""
    params.update(sides=_check_sides(sides), length=length, outside=outside)
    return make_layer(
        GeomRug(),
        mapping,
        data,
        stat=stat,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
