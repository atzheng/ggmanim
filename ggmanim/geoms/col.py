"""geom_col (and its alias geom_bar): bars with heights from the data."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from manim import Polygon, Square, VGroup, VMobject

from ..registry import register
from ..theme import Theme, theme_default
from .base import KEY_SIZE, Geom, finite_rows, make_layer, num, resolve_color, transform_points

__all__ = ["GeomCol", "geom_col", "geom_bar", "resolution"]


def resolution(x) -> float:
    """Smallest gap between distinct finite values of ``x`` (1 if fewer than two)."""
    x = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce"), dtype=float)
    u = np.unique(x[np.isfinite(x)])
    if len(u) < 2:
        return 1.0
    d = np.diff(u)
    d = d[d > 1e-9 * max(1.0, float(np.abs(u).max()))]
    return float(d.min()) if len(d) else 1.0


class GeomCol(Geom):
    """Bars. ``draw_panel`` returns a ``VGroup`` with one ``Polygon`` per (finite)
    row, in data order.

    :meth:`setup_data` (run by the pipeline after the stat, in transformed space)
    adds ``xmin = x - width/2``, ``xmax = x + width/2``, ``ymin = min(y, 0)``,
    ``ymax = max(y, 0)``. ``width`` (param or column, transformed x units)
    defaults to ``0.9 * resolution(x)``; so on ``scale_x_log10`` bars have equal
    widths on screen, as in ggplot2. The base ``0`` is in transformed space (data
    value 1 on a log10 y scale, like ggplot2), and both ends are clipped to the
    panel's y range so bars never extend past the axes.

    The four corners go through ``coord.transform``, so bars are correct under
    any scale transform. ``color`` (outline) defaults to ``None`` = no outline.
    """

    required_aes = ("x", "y")
    default_aes = {"color": None, "fill": "#595959", "linewidth": 1.0, "alpha": 1.0,
                   "linetype": "solid"}
    extra_params = ("width",)

    def setup_data(self, data: pd.DataFrame, params) -> pd.DataFrame:
        data = data.copy()
        if len(data) == 0:
            for c in ("xmin", "xmax", "ymin", "ymax"):
                data[c] = pd.Series(dtype=float)
            return data
        x = pd.to_numeric(data["x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(data["y"], errors="coerce").to_numpy(dtype=float)
        width = params.get("width") if params else None
        if width is None and "width" in data.columns:
            width = pd.to_numeric(data["width"], errors="coerce").to_numpy(dtype=float)
        if width is None:
            width = 0.9 * resolution(x)
        width = np.broadcast_to(np.asarray(width, dtype=float), x.shape)
        data["xmin"] = x - width / 2
        data["xmax"] = x + width / 2
        data["ymin"] = np.minimum(y, 0.0)
        data["ymax"] = np.maximum(y, 0.0)
        return data

    def draw_panel(self, data, panel_params, coord, theme: Optional[Theme] = None,
                   width=None, **params) -> VMobject:
        theme = theme if theme is not None else theme_default()
        self.check_required_aes(data)
        if not {"xmin", "xmax", "ymin", "ymax"} <= set(data.columns):
            data = self.setup_data(data, {"width": width})
        data = finite_rows(self.use_defaults(data, params), ("xmin", "xmax", "ymin", "ymax"))
        xmin = data["xmin"].to_numpy(float)
        xmax = data["xmax"].to_numpy(float)
        ymin = data["ymin"].to_numpy(float)
        ymax = data["ymax"].to_numpy(float)
        if panel_params is not None:
            lo, hi = panel_params.y_range
            ymin, ymax = np.clip(ymin, lo, hi), np.clip(ymax, lo, hi)
        # one vectorised transform for all corners: (n, 4, 3)
        cx = np.stack([xmin, xmax, xmax, xmin], axis=1)
        cy = np.stack([ymin, ymin, ymax, ymax], axis=1)
        corners = transform_points(coord, cx.ravel(), cy.ravel()).reshape(-1, 4, 3)
        bars = VGroup()
        for c, row in zip(corners, data.to_dict("records")):
            bars.add(self._bar(Polygon(*c), row, theme))
        return bars

    @staticmethod
    def _bar(poly: VMobject, row, theme) -> VMobject:
        alpha = num(row.get("alpha"), 1.0)
        poly.set_fill(resolve_color(row.get("fill"), theme.get_foreground()), opacity=alpha)
        outline = resolve_color(row.get("color"))
        if outline is None:
            poly.set_stroke(width=0, opacity=0)
        else:
            poly.set_stroke(color=outline, width=num(row.get("linewidth"), 1.0), opacity=alpha)
        return poly

    def draw_key(self, data_row, theme: Optional[Theme] = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        return self._bar(Square(side_length=KEY_SIZE), self.key_aes(data_row), theme)


@register
def geom_col(mapping=None, data=None, *, stat="identity", position="identity",
             inherit_aes: bool = True, name: Optional[str] = None, **params):
    """Bars whose heights are the data's ``y``. ``width=`` in x (transformed) units;
    constant ``fill``/``color``/``alpha`` go in ``params``."""
    return make_layer(GeomCol(), mapping, data, stat=stat, position=position,
                      inherit_aes=inherit_aes, name=name, params=params)


@register
def geom_bar(mapping=None, data=None, *, stat="identity", position="identity",
             inherit_aes: bool = True, name: Optional[str] = None, **params):
    """Alias of :func:`geom_col` (identity heights) until ``stat_count`` lands
    in Phase 3, at which point it will default to counting rows."""
    return make_layer(GeomCol(), mapping, data, stat=stat, position=position,
                      inherit_aes=inherit_aes, name=name, params=params)
