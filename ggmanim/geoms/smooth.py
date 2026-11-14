"""geom_smooth: fitted line (stat_smooth) plus a confidence ribbon."""

from __future__ import annotations

import numpy as np
from manim import LEFT, RIGHT, Line, Polygon, Square, VGroup, VMobject

from ..registry import register
from ..theme import Theme, theme_default
from ._clip import clip_polygon, panel_rect
from ._linetype import apply_linetype
from ._segments import segmented_path, varies_within
from .base import (
    KEY_SIZE,
    assemble,
    attach_marks,
    clipped_path_points,
    finite_rows,
    first,
    group_label,
    make_label,
    make_layer,
    multi_path_mobject,
    num,
    resolve_color,
    transform_points,
)
from .line import GeomLine

__all__ = ["GeomSmooth", "geom_smooth"]


class GeomSmooth(GeomLine):
    """Smoother: one mobject per group, in ``group`` order.

    Each group's mobject is the fitted line (a ``VMobject``) or, when ``se`` is
    true and the data has ``ymin``/``ymax``, ``VGroup(ribbon, line)`` with the
    ribbon (``Polygon``) filled with ``fill`` at opacity ``alpha`` behind the line.
    ``alpha`` applies to the ribbon only (as in ggplot2). ``ymin``/``ymax`` train
    the y scale. When ``color``/``linewidth`` vary along a group (e.g. mapped with
    ``after_stat``), the line is drawn per segment like :class:`GeomPath`.
    """

    default_stat = "smooth"
    default_aes = {
        "color": "#3366FF",
        "fill": "#999999",
        "linewidth": 3.0,
        "alpha": 0.4,
        "linetype": "solid",
    }
    extra_params: tuple[str, ...] = ("label", "se")

    def draw_panel(
        self, data, panel_params, coord, theme: Theme | None = None, label=None, se=True, **params
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        self.check_required_aes(data)
        data = finite_rows(self.use_defaults(data, params), ("x", "y"))
        fg = theme.get_foreground()
        groups, labels = VGroup(), VGroup()
        for i, (gid, gdf) in enumerate(data.groupby("group", sort=True)):
            gdf = gdf.sort_values("x", kind="stable")
            x = gdf["x"].to_numpy(float)
            pieces = clipped_path_points(coord, panel_params, x, gdf["y"].to_numpy(float))
            color = label_color = resolve_color(first(gdf, "color"), fg)
            line: VMobject
            if varies_within(gdf, ("color", "linewidth")):
                # per-segment fit line (alpha is the ribbon's, the line stays opaque)
                line = segmented_path(coord, panel_params, gdf.assign(alpha=1.0), fg)
                if len(line):
                    label_color = line[-1].get_stroke_color()
            else:
                # clip first, then dash each visible subpath
                line = multi_path_mobject(pieces, color, num(first(gdf, "linewidth"), 3.0), 1.0)
                line = apply_linetype(line, first(gdf, "linetype"))
            mob = line
            if se and {"ymin", "ymax"} <= set(gdf.columns):
                ymin = gdf["ymin"].to_numpy(float)
                ymax = gdf["ymax"].to_numpy(float)
                ok = np.isfinite(ymin) & np.isfinite(ymax)
                rx = np.concatenate([x[ok], x[ok][::-1]])
                ry = np.concatenate([ymax[ok], ymin[ok][::-1]])
                rect = panel_rect(coord, panel_params)
                if rect is not None and ok.sum() >= 2:
                    rx, ry = clip_polygon(rx, ry, rect)
                if ok.sum() >= 2 and len(rx) >= 3:
                    ribbon = Polygon(*transform_points(coord, rx, ry))
                    ribbon.set_fill(
                        resolve_color(first(gdf, "fill"), color),
                        opacity=num(first(gdf, "alpha"), 0.4),
                    )
                    ribbon.set_stroke(width=0, opacity=0)
                    mob = VGroup(ribbon, line)
            groups.add(mob)
            lab = make_label(theme, group_label(label, i, gid), label_color)
            if lab is not None and pieces:
                labels.add(lab.next_to(pieces[-1][-1], RIGHT, buff=0.1))
        return attach_marks(assemble(groups, labels), data, per="group")

    def draw_key(self, data_row, theme: Theme | None = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        row = self.key_aes(data_row)
        color = resolve_color(row.get("color"), theme.get_foreground())
        w = getattr(theme, "legend_key_width", None) or KEY_SIZE  # wider for linetype legends
        bg = Square(side_length=KEY_SIZE, stroke_width=0).stretch_to_fit_width(w)
        bg.set_fill(resolve_color(row.get("fill"), color), opacity=num(row.get("alpha"), 0.4))
        line = Line(
            LEFT * w / 2,
            RIGHT * w / 2,
            color=color,
            stroke_width=num(row.get("linewidth"), 3.0),
        )
        return VGroup(bg, apply_linetype(line, row.get("linetype"), max_period=w / 2))


@register
def geom_smooth(
    mapping=None,
    data=None,
    *,
    method: str = "lm",
    se: bool = True,
    stat="smooth",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Smoothed conditional mean with a confidence ribbon (``se=True``).

    Stat params (``degree``, ``n``, ``level``, ``fullrange``) and constant
    aesthetics (``color``, ``fill``, ``alpha``, ``linewidth``) go in ``params``.
    """
    params.update(method=method, se=se)
    return make_layer(
        GeomSmooth(),
        mapping,
        data,
        stat=stat,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
