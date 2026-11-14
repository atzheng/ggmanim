"""geom_abline: reference lines ``y = intercept + slope * x`` across the panel.

Like ``geom_vline``/``geom_hline`` the layer carries its own data when given
``slope=``/``intercept=`` and never inherits the plot's aesthetics. It trains no
position scale (ggplot2 does the same: an abline adapts to the panel, it does
not extend it). Each line is evaluated at the panel's x limits in *transformed*
space and clipped to the panel, so on a log scale the line is straight on
screen and ``slope``/``intercept`` are in transformed units (ggplot2's
behaviour: ``scale_x_log10`` + ``geom_abline(slope=1)`` is ``log10 y = log10 x``).
Under a non-linear coord (``coord_trans``, ``coord_polar``) the line is
munched, so it bends with the coord.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from manim import DL, UR, Line, VGroup, VMobject

from ..aes import aes
from ..registry import register
from ..theme import Theme, theme_default
from ._clip import clip_segments, panel_rect
from ._linetype import apply_linetype
from ._segments import munched_segment
from .base import (
    KEY_SIZE,
    Geom,
    attach_marks,
    finite_rows,
    make_layer,
    multi_path_mobject,
    num,
    resolve_color,
    transform_points,
)

__all__ = ["GeomAbline", "geom_abline"]


class GeomAbline(Geom):
    """Lines ``y = intercept + slope * x``: one ``Line`` per (visible) row
    spanning the panel's x range (clipped where it leaves through the top or
    bottom). Marks are per row."""

    required_aes = ("slope", "intercept")
    default_aes = {"color": None, "linewidth": 2.0, "alpha": 1.0, "linetype": "solid"}
    x_aes: tuple[str, ...] = ()
    y_aes: tuple[str, ...] = ()
    na_aes: tuple[str, ...] | None = ("slope", "intercept")

    def draw_panel(
        self, data, panel_params, coord, theme: Theme | None = None, **params
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        self.check_required_aes(data)
        data = finite_rows(self.use_defaults(data, params), self.required_aes)
        fg = theme.get_foreground()
        slope = data["slope"].to_numpy(float)
        icpt = data["intercept"].to_numpy(float)
        x0, x1 = (float(v) for v in panel_params.x_range)
        y0, y1 = icpt + slope * x0, icpt + slope * x1
        if not getattr(coord, "is_linear", True):
            return self._draw_munched(data, panel_params, coord, fg, x0, x1, y0, y1)
        rect = panel_rect(coord, panel_params)
        if rect is None:
            t0, t1 = np.zeros(len(data)), np.ones(len(data))
            vis = np.ones(len(data), dtype=bool)
        else:
            t0, t1, vis = clip_segments(x0, y0, x1, y1, rect)
        a = transform_points(coord, x0 + t0 * (x1 - x0), y0 + t0 * (y1 - y0))
        b = transform_points(coord, x0 + t1 * (x1 - x0), y0 + t1 * (y1 - y0))
        lines, drawn = VGroup(), []
        for i, row in enumerate(data.to_dict("records")):
            if not vis[i] or np.allclose(a[i], b[i]):
                continue
            line = Line(
                a[i],
                b[i],
                color=resolve_color(row.get("color"), fg),
                stroke_width=num(row.get("linewidth"), 2.0),
                stroke_opacity=num(row.get("alpha"), 1.0),
            )
            lines.add(apply_linetype(line, row.get("linetype")))
            drawn.append(i)
        return attach_marks(lines, data.iloc[drawn], per="row")

    def _draw_munched(self, data, panel_params, coord, fg, x0, x1, y0, y1) -> VMobject:
        """Non-linear coords: each line munched into a curve (ggplot2 draws an
        abline as a segment, which ``coord_munch`` bends)."""
        lines, drawn = VGroup(), []
        for i, row in enumerate(data.to_dict("records")):
            pieces, _, _ = munched_segment(coord, panel_params, x0, y0[i], x1, y1[i])
            if not pieces:
                continue
            path = multi_path_mobject(
                pieces,
                resolve_color(row.get("color"), fg),
                num(row.get("linewidth"), 2.0),
                num(row.get("alpha"), 1.0),
            )
            lines.add(apply_linetype(path, row.get("linetype")))
            drawn.append(i)
        return attach_marks(lines, data.iloc[drawn], per="row")

    def draw_key(self, data_row, theme: Theme | None = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        row = self.key_aes(data_row)
        line = Line(
            DL * KEY_SIZE / 2,
            UR * KEY_SIZE / 2,
            color=resolve_color(row.get("color"), theme.get_foreground()),
            stroke_width=num(row.get("linewidth"), 2.0),
            stroke_opacity=num(row.get("alpha"), 1.0),
        )
        return apply_linetype(line, row.get("linetype"), max_period=KEY_SIZE / 2)


@register
def geom_abline(
    mapping=None,
    data=None,
    *,
    slope=None,
    intercept=None,
    position="identity",
    name: str | None = None,
    **params,
):
    """Line(s) ``y = intercept + slope * x`` (numbers or equal-length lists, in
    transformed units) spanning the panel.

    With neither ``slope``/``intercept`` nor a mapping, draws ``y = x``; a
    missing ``slope`` defaults to 1 and a missing ``intercept`` to 0. With
    ``slope=``/``intercept=`` the layer carries its own data and never inherits
    the plot's aesthetics; alternatively map ``aes(slope=..., intercept=...)``.
    """
    if slope is not None or intercept is not None or (mapping is None and data is None):
        if mapping is not None or data is not None:
            raise TypeError("pass either slope=/intercept= or mapping/data, not both")
        s = np.atleast_1d(np.asarray(1.0 if slope is None else slope, dtype=float))
        i = np.atleast_1d(np.asarray(0.0 if intercept is None else intercept, dtype=float))
        s, i = np.broadcast_arrays(s, i)
        data = pd.DataFrame({"slope": s, "intercept": i})
        mapping = aes(slope="slope", intercept="intercept")
    return make_layer(
        GeomAbline(),
        mapping,
        data,
        stat="identity",
        position=position,
        inherit_aes=False,
        name=name,
        params=params,
    )
