"""geom_violin: mirrored density outlines from stat_ydensity.

Each group (one violin) is drawn as one filled polygon: the density curve
``x - violinwidth * (x - xmin)`` going up and ``x + violinwidth * (xmax - x)``
coming back down, so the violin is symmetric about ``x`` and at most ``width``
wide (``0.9 * resolution(x)`` by default; ``0.9`` on a discrete scale). The
default ``position="dodge"`` puts violins that share an x side by side.

``draw_quantiles=[...]`` (ggplot2 3.x's argument) computes those quantiles in the
stat and draws a horizontal line across the violin at each; with ggplot2 4.0's
``quantile_linetype`` (and ``quantile_color``/``quantile_linewidth``) the
default quantiles (0.25, 0.5, 0.75) are drawn. ``fill`` defaults to none,
``color`` to the theme foreground; ``alpha`` applies to the fill only.

Horizontal violins (``flipped_aes``, ``orientation="y"``) are drawn with x and y
swapped. A mark is the group's polygon, or ``VGroup(polygon, quantile_lines)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from manim import VGroup, VMobject

from ..registry import register
from ..stats.boxplot import flip_names, resolution, resolve_flipped
from ..theme import Theme, theme_default
from ._clip import clip_polygon, panel_rect
from ._rect import filled_key, rings_mobject, style_filled
from .base import Geom, attach_marks, first, make_layer, multi_path_mobject, num, transform_points
from .boxplot import oriented
from .interval import _clipped_segments, _flip_panel, _style

__all__ = ["GeomViolin", "geom_violin"]


def _is_flipped(data: pd.DataFrame, params: Mapping[str, Any]) -> bool:
    return resolve_flipped(data, None, params.get("orientation"))


class GeomViolin(Geom):
    """Violins: one mark per group (see the module docstring)."""

    required_aes: tuple[str, ...] = ("x", "y", "violinwidth")
    default_aes = {
        "color": None,
        "fill": None,
        "linewidth": 2.0,
        "alpha": 1.0,
        "linetype": "solid",
    }
    extra_params: tuple[str, ...] = (
        "width",
        "orientation",
        "draw_quantiles",
        "quantile_color",
        "quantile_linetype",
        "quantile_linewidth",
    )
    default_stat = "ydensity"
    default_position = "dodge"
    na_aes: tuple[str, ...] | None = ("x", "y")

    def check_required_aes(self, data: pd.DataFrame, flipped: bool = False) -> None:
        cols = set(flip_names(data).columns if flipped else data.columns)
        missing = [a for a in self.required_aes if a not in cols]
        if missing:
            raise ValueError(
                "GeomViolin requires the following missing aesthetics: " + ", ".join(missing)
            )

    def setup_data(self, data: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        params = params or {}
        flipped = _is_flipped(data, params)
        data = flip_names(data) if flipped else data.copy()
        data["flipped_aes"] = flipped
        if len(data) and "x" in data.columns:
            x = pd.to_numeric(data["x"], errors="coerce").to_numpy(dtype=float)
            if "width" in data.columns:
                width = pd.to_numeric(data["width"], errors="coerce").to_numpy(dtype=float)
            elif params.get("width") is not None:
                width = np.full(len(data), float(params["width"]))
            else:
                discrete = params.get("_y_discrete" if flipped else "_x_discrete")
                width = np.full(len(data), 0.9 * (1.0 if discrete else resolution(x)))
            data["xmin"] = x - width / 2
            data["xmax"] = x + width / 2
            data = data.drop(columns=["width"], errors="ignore")
        return flip_names(data) if flipped else data

    def draw_panel(
        self, data, panel_params, coord, theme: Theme | None = None, **params
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        if not len(data):
            return attach_marks(VGroup(), data, per="group")
        flipped = _is_flipped(data, params)
        self.check_required_aes(data, flipped)
        if not {"xmin", "xmax"} <= set((flip_names(data) if flipped else data).columns):
            data = self.setup_data(data, params)
        data = self.use_defaults(data, params)
        if flipped:
            data = flip_names(data)
            coord, panel_params = _flip_panel(coord, panel_params)
        rect = panel_rect(coord, panel_params)
        fg = theme.get_foreground()
        qlinetype = params.get("quantile_linetype")
        show_q = "quantile" in data.columns and (
            params.get("draw_quantiles") is not None
            or (qlinetype is not None and qlinetype not in ("blank", 0))
        )
        out, drawn = VGroup(), []
        for gid, gdf in data.groupby("group", sort=True):
            gdf = gdf.sort_values("y", kind="stable")
            g = {
                c: pd.to_numeric(gdf[c], errors="coerce").to_numpy(dtype=float)
                for c in ("x", "y", "xmin", "xmax", "violinwidth")
            }
            ok = np.isfinite(g["x"]) & np.isfinite(g["y"]) & np.isfinite(g["violinwidth"])
            if ok.sum() < 2:
                continue
            x, y, vw = g["x"][ok], g["y"][ok], g["violinwidth"][ok]
            left = x - vw * (x - g["xmin"][ok])
            right = x + vw * (g["xmax"][ok] - x)
            px = np.concatenate([left, right[::-1]])
            py = np.concatenate([y, y[::-1]])
            if rect is not None:
                px, py = clip_polygon(px, py, rect)
            style = {
                "fill": first(gdf, "fill"),
                "alpha": first(gdf, "alpha"),
                "color": first(gdf, "color", fg),
                "linewidth": first(gdf, "linewidth"),
                "linetype": first(gdf, "linetype"),
            }
            if style["fill"] is None:
                style["alpha"] = 0.0
            ring = [transform_points(coord, px, py)] if len(px) >= 3 else []
            mob: VMobject = style_filled(rings_mobject(ring), style, theme, 2.0)
            if show_q:
                lines = self._quantiles(gdf[ok], left, right, coord, rect, style, fg, params)
                if lines is not None:
                    mob = VGroup(mob, lines)
            out.add(mob)
            drawn.append(gid)
        return attach_marks(out, data[data["group"].isin(drawn)], per="group")

    def _quantiles(self, gdf, left, right, coord, rect, style, fg, params) -> VMobject | None:
        q = pd.to_numeric(gdf["quantile"], errors="coerce").to_numpy(dtype=float)
        sel = np.isfinite(q)
        if not sel.any():
            return None
        y = gdf["y"].to_numpy(dtype=float)[sel]
        pieces = _clipped_segments(coord, rect, left[sel], y, right[sel], y)
        if not pieces:
            return None
        row = {
            "color": params.get("quantile_color") or style["color"],
            "alpha": 1.0,
            "linetype": params.get("quantile_linetype") or style["linetype"],
        }
        lw = num(params.get("quantile_linewidth"), num(style["linewidth"], 2.0))
        return _style(multi_path_mobject(pieces, None, lw, 1.0), row, fg, lw)

    def draw_key(self, data_row, theme: Theme | None = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        row = self.key_aes(data_row)
        if row.get("color") is None:
            row["color"] = theme.get_foreground()
        if row.get("fill") is None:
            row["alpha"] = 0.0
        return filled_key(row, theme, default_linewidth=2.0)


@register
def geom_violin(
    mapping=None,
    data=None,
    *,
    stat="ydensity",
    position="dodge",
    draw_quantiles=None,
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Violin plots (ggplot2's ``geom_violin``); see :mod:`ggmanim.geoms.violin`.

    Stat params: ``scale`` (``"area"``, ``"count"``, ``"width"``), ``trim``
    (``True``), ``bw``, ``adjust``, ``kernel``, ``bounds``, ``quantiles``. Geom
    params: ``width``, ``draw_quantiles`` (a list of probabilities to draw),
    ``quantile_color``/``quantile_linetype``/``quantile_linewidth``,
    ``orientation``. Default position ``"dodge"``."""
    if draw_quantiles is not None:
        params["draw_quantiles"] = list(np.atleast_1d(draw_quantiles))
        params.setdefault("quantiles", params["draw_quantiles"])
    return make_layer(
        GeomViolin(),
        mapping,
        data,
        stat=stat,
        position=oriented(position),
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
