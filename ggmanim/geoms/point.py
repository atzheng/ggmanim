"""geom_point: scatter plot (one ``Dot`` per row)."""

from __future__ import annotations

import numpy as np
from manim import ORIGIN, VGroup, VMobject

from ..registry import register
from ..theme import Theme, theme_default
from ._batch import draw_points_fast, use_fast
from ._clip import panel_rect
from ._marker_clip import clip_marker, marker_extent, near_box, scene_box
from ._markers import make_marker
from .base import (
    Geom,
    attach_marks,
    color_resolver,
    finite_rows,
    make_layer,
    num,
    num_array,
    resolve_color,
    transform_points,
)

__all__ = ["GeomPoint", "geom_point"]


class GeomPoint(Geom):
    """Points: ``draw_panel`` returns a ``VGroup`` with one marker per (finite) row,
    in data order (``.marks``/``.keys``: one per drawn row, see
    :func:`~ggmanim.geoms.base.attach_marks`).

    ``size`` is the dot radius in scene units (other shapes have the same area);
    ``color=None`` is the theme foreground; ``fill`` (if set) colours solid
    markers instead of ``color``. ``shape`` (``None`` = circle) selects the
    marker, see :mod:`ggmanim.geoms._markers` and
    :func:`ggmanim.scales.shape.parse_shape`.

    ``fast`` (``None``: above ``theme.fast_threshold`` rows) draws batched marks
    instead, one ``VMobject`` per style (:mod:`ggmanim.geoms._batch`; no
    per-mark ``.keys``).
    """

    required_aes = ("x", "y")
    default_aes = {"color": None, "fill": None, "size": 0.06, "alpha": 1.0, "shape": None}
    extra_params: tuple[str, ...] = ("fast",)

    def draw_panel(
        self, data, panel_params, coord, theme: Theme | None = None, *, fast=None, **params
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        self.check_required_aes(data)
        if use_fast(fast, data, theme, type(self).__name__):
            return draw_points_fast(
                self.use_defaults(data, params),
                panel_params,
                coord,
                theme,
                lambda rows: self.draw_panel(
                    rows, panel_params, coord, theme, fast=False, **params
                ),
            )
        data = finite_rows(self.use_defaults(data, params), ("x", "y"))
        rect = panel_rect(coord, panel_params)
        fg = theme.get_foreground()
        pts = transform_points(coord, data["x"].to_numpy(float), data["y"].to_numpy(float))
        radius = num_array(data["size"], 0.06)
        box = None
        inside = None
        if rect is not None:  # clip="on": markers are clipped to the panel
            box = scene_box(coord, rect)
            ext = marker_extent(radius)
            keep = near_box(pts, box, ext)
            data, pts, radius = data[keep], pts[keep], radius[keep]
            inside = near_box(pts, box, -ext[keep])  # surely unclipped: skip clip_marker
        color_of, fill_of = color_resolver(fg), color_resolver()
        alpha = num_array(data["alpha"], 1.0)
        if inside is None:
            inside = np.ones(len(pts), dtype=bool)
        rows = zip(
            pts,
            data["shape"].tolist(),
            radius,
            data["color"].tolist(),
            data["fill"].tolist(),
            alpha,
            inside.tolist(),
        )
        marks = []
        drawn = []
        for i, (p, shape, r, color, fill, a, ok) in enumerate(rows):
            color = color_of(color)
            dot = make_marker(
                shape,
                p,
                radius=float(r),
                color=color if color is not None else fg,
                fill=fill_of(fill),
                alpha=float(a),
            )
            shown = dot if ok or box is None else clip_marker(dot, box)  # None: clipped away
            if shown is not None:
                marks.append(shown)
                drawn.append(i)
        dots = VGroup(*marks)  # one add: VGroup.add per mark is quadratic
        return attach_marks(dots, data.iloc[drawn], per="row")

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
    ``size=0.08``, ``alpha=0.5``) go in ``params``. ``fast=True`` draws batched
    marks (default ``None``: when the panel has more than
    ``theme.fast_threshold`` rows), see :mod:`ggmanim.geoms._batch`."""
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
