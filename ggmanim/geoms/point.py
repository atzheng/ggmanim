"""geom_point: scatter plot (one ``Dot`` per row)."""

from __future__ import annotations

from manim import ORIGIN, VGroup, VMobject

from ..registry import register
from ..theme import Theme, theme_default
from ._clip import inside, panel_rect
from ._markers import make_marker
from .base import Geom, finite_rows, make_layer, num, resolve_color, transform_points

__all__ = ["GeomPoint", "geom_point"]


class GeomPoint(Geom):
    """Points: ``draw_panel`` returns a ``VGroup`` with one marker per (finite) row,
    in data order.

    ``size`` is the dot radius in scene units (other shapes have the same area);
    ``color=None`` is the theme foreground; ``fill`` (if set) colours solid
    markers instead of ``color``. ``shape`` (``None`` = circle) selects the
    marker, see :mod:`ggmanim.geoms._markers` and
    :func:`ggmanim.scales.shape.parse_shape`.
    """

    required_aes = ("x", "y")
    default_aes = {"color": None, "fill": None, "size": 0.06, "alpha": 1.0, "shape": None}

    def draw_panel(
        self, data, panel_params, coord, theme: Theme | None = None, **params
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        self.check_required_aes(data)
        data = finite_rows(self.use_defaults(data, params), ("x", "y"))
        rect = panel_rect(coord, panel_params)
        if rect is not None:  # clip="on": points outside the panel are not drawn
            data = data[inside(data["x"].to_numpy(float), data["y"].to_numpy(float), rect)]
        fg = theme.get_foreground()
        pts = transform_points(coord, data["x"].to_numpy(float), data["y"].to_numpy(float))
        dots = VGroup()
        for p, row in zip(pts, data.to_dict("records")):
            dots.add(self._dot(p, row, fg))
        return dots

    @staticmethod
    def _dot(point, row, fg) -> VMobject:
        color = resolve_color(row.get("color"), fg)
        return make_marker(
            row.get("shape"),
            point,
            radius=num(row.get("size"), 0.06),
            color=color if color is not None else fg,
            fill=resolve_color(row.get("fill")),
            alpha=num(row.get("alpha"), 1.0),
        )

    def draw_key(self, data_row, theme: Theme | None = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        return self._dot(ORIGIN, self.key_aes(data_row), theme.get_foreground())


@register
def geom_point(
    mapping=None,
    data=None,
    *,
    stat="identity",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Scatter plot: one dot per row. Constant aesthetics (``color=RED``,
    ``size=0.08``, ``alpha=0.5``) go in ``params``."""
    return make_layer(
        GeomPoint(),
        mapping,
        data,
        stat=stat,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
