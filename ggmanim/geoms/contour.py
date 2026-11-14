"""geom_contour / geom_contour_filled: contour lines and filled bands of
``aes(x, y, z)`` (stats :class:`~ggmanim.stats.contour.StatContour` and
:class:`~ggmanim.stats.contour.StatContourFilled`).

Filled bands are polygons whose rings (``subgroup``) come from the stat already
oriented for the *nonzero* fill rule (outer boundaries counter-clockwise, holes
clockwise), so :class:`GeomContourFilled` draws them as given instead of
re-orienting them by nesting like :class:`~ggmanim.geoms.polygon.GeomPolygon`
(contour rings may share border edges, which defeats point-in-polygon nesting).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from manim import VGroup, VMobject

from ..registry import register
from ..theme import Theme, theme_default
from ._clip import clip_polygon, panel_rect
from ._rect import rings_mobject, seal_seams, style_filled
from .base import attach_marks, transform_points
from .count import stat_layer
from .line import GeomPath
from .polygon import GeomPolygon

__all__ = [
    "GeomContour",
    "GeomContourFilled",
    "draw_oriented_polygons",
    "geom_contour",
    "geom_contour_filled",
]


class GeomContour(GeomPath):
    """Contour lines: a :class:`~ggmanim.geoms.line.GeomPath` (one path per
    contour piece) fed by ``stat_contour``. Colour defaults to ``#3366FF``."""

    default_stat = "contour"
    default_aes = {"color": "#3366FF", "linewidth": 2.0, "alpha": 1.0, "linetype": "solid"}

    def aesthetics(self) -> frozenset:
        return super().aesthetics() | {"z", "weight"}


def draw_oriented_polygons(
    geom, data: pd.DataFrame, panel_params, coord, theme: Theme, params
) -> VMobject:
    """One filled mobject per group whose ``subgroup`` rings keep their
    orientation (nonzero fill). Rings are clipped to the panel one by one; opaque
    bands without an outline get a fill-coloured hairline so no anti-aliasing
    seams show between neighbouring bands (:func:`~._rect.seal_seams`)."""
    geom.check_required_aes(data)
    data = geom.use_defaults(data, params)
    rect = panel_rect(coord, panel_params)
    has_sub = "subgroup" in data.columns
    out = VGroup()
    drawn = []
    for gid, gdf in data.groupby("group", sort=True):
        parts = [p for _, p in gdf.groupby("subgroup", sort=False)] if has_sub else [gdf]
        rings = []
        for part in parts:
            x = pd.to_numeric(part["x"], errors="coerce").to_numpy(float)
            y = pd.to_numeric(part["y"], errors="coerce").to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            if rect is not None:
                x, y = clip_polygon(x, y, rect)
            if len(x) >= 3:
                rings.append(transform_points(coord, x, y))
        if not rings:
            continue
        mob = style_filled(rings_mobject(rings), gdf.iloc[0].to_dict(), theme)
        out.add(mob if isinstance(mob, VGroup) else seal_seams(mob))
        drawn.append(gid)
    return attach_marks(out, data[data["group"].isin(drawn)], per="group")


class GeomContourFilled(GeomPolygon):
    """Filled contour bands: one mobject per band (group), its rings as
    subpaths, ``fill`` mapped to ``after_stat("level")`` by default."""

    default_stat = "contour_filled"

    def aesthetics(self) -> frozenset:
        return super().aesthetics() | {"z", "weight"}

    def draw_panel(
        self, data, panel_params, coord, theme: Theme | None = None, **params
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        return draw_oriented_polygons(self, data, panel_params, coord, theme, params)


def _layer(geom, stat, mapping, data, position, inherit_aes, name, params: dict[str, Any]):
    return stat_layer(
        stat,
        geom,
        mapping,
        data,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )


@register
def geom_contour(
    mapping=None,
    data=None,
    *,
    stat="contour",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Contour lines of a gridded ``aes(x, y, z)``. Levels from ``bins``,
    ``binwidth`` or ``breaks`` (default ``pretty`` over the z range); map
    ``color=after_stat("level")`` to colour by level."""
    return _layer(GeomContour(), stat, mapping, data, position, inherit_aes, name, params)


@register
def geom_contour_filled(
    mapping=None,
    data=None,
    *,
    stat="contour_filled",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Filled contour bands of a gridded ``aes(x, y, z)``, ``fill`` mapped to the
    band (``after_stat("level")``, an ordered category ``"(lo, hi]"``)."""
    return _layer(GeomContourFilled(), stat, mapping, data, position, inherit_aes, name, params)
