"""geom_polygon: filled polygons, one per group, with optional holes (``subgroup``)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from manim import VGroup, VMobject

from ..registry import register
from ..theme import Theme, theme_default
from ._clip import clip_polygon, panel_rect
from ._rect import filled_key, orient_rings, rings_mobject, style_filled
from .base import Geom, attach_marks, make_layer, munch_path, transform_points

__all__ = ["GeomPolygon", "geom_polygon"]


class GeomPolygon(Geom):
    """Polygons: the rows of each group, in data order, are the vertices of one
    closed polygon (the closing edge is implied).

    ``draw_panel`` returns a ``VGroup`` with one ``VMobject`` per group with at
    least three visible vertices, in ``group`` order (``.marks``/``.keys``: one per
    drawn group). Styling (``fill``, ``alpha`` (fill only), ``color`` outline,
    ``linewidth``, ``linetype``) comes from the group's first row.

    A mapped ``subgroup`` splits a group into rings, in order of appearance: the
    first is usually the outline and the others holes. All rings are subpaths of
    the one mobject and are filled with the *even-odd* rule, as in ggplot2. Manim
    has no fill-rule setting (it always uses nonzero winding), so the rings are
    re-oriented by nesting depth (:func:`~._rect.orient_rings`), which gives the
    same result for rings that do not cross each other.

    Each ring is clipped to the panel separately (Sutherland-Hodgman), so with an
    outline the clipped edge along the panel border is stroked too.
    """

    required_aes = ("x", "y")
    default_aes: dict[str, Any] = {
        "color": None,
        "fill": "#595959",
        "linewidth": 1.0,
        "alpha": 1.0,
        "linetype": "solid",
    }

    def aesthetics(self) -> frozenset:
        return super().aesthetics() | {"subgroup"}

    def draw_panel(
        self, data, panel_params, coord, theme: Theme | None = None, **params
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        self.check_required_aes(data)
        data = self.use_defaults(data, params)
        rect = panel_rect(coord, panel_params)
        has_sub = "subgroup" in data.columns
        out = VGroup()
        drawn_groups = []
        for gid, gdf in data.groupby("group", sort=True):
            parts = (
                [p for _, p in gdf.groupby("subgroup", sort=False, dropna=False)]
                if has_sub
                else [gdf]
            )
            rings = []
            for part in parts:
                x = pd.to_numeric(part["x"], errors="coerce").to_numpy(float)
                y = pd.to_numeric(part["y"], errors="coerce").to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                x, y = munch_path(coord, x[ok], y[ok], closed=True)
                if rect is not None:
                    x, y = clip_polygon(x, y, rect)
                if len(x) >= 3:
                    rings.append(transform_points(coord, x, y))
            if not rings:
                continue
            row = gdf.iloc[0].to_dict()
            out.add(style_filled(rings_mobject(orient_rings(rings)), row, theme))
            drawn_groups.append(gid)
        return attach_marks(out, data[data["group"].isin(drawn_groups)], per="group")

    def draw_key(self, data_row, theme: Theme | None = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        return filled_key(self.key_aes(data_row), theme)


@register
def geom_polygon(
    mapping=None,
    data=None,
    *,
    stat="identity",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Filled polygons through ``aes(x, y)``, one per ``group``; map ``subgroup``
    to cut holes (even-odd fill). Constant ``fill``, ``color`` (outline), ``alpha``
    (fill only), ``linewidth`` and ``linetype`` go in ``params``."""
    return make_layer(
        GeomPolygon(),
        mapping,
        data,
        stat=stat,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
