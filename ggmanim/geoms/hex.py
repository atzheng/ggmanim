"""geom_hex: hexagonal 2-D bin counts (``stat_bin_hex``) drawn as hexagons.

Each hexagon is the pointy-topped hexagon of :func:`~ggmanim.stats.hex.hexagon_vertices`
in transformed (data) units, put through ``coord.transform`` -- so neighbouring
hexagons share their edges exactly (no gaps) whatever the panel's aspect ratio,
and the tiles follow ``coord_flip``. On a panel whose x and y units have the same
scene length per bin width the hexagons are regular.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from manim import VGroup, VMobject

from ..registry import register
from ..stats.hex import hexagon_vertices
from ..theme import Theme, theme_default
from ._clip import clip_polygon, panel_rect
from ._rect import filled_key, ring_points, seal_seams, style_filled
from .base import Geom, attach_marks, finite_rows, transform_points
from .col import resolution
from .contour import _layer

__all__ = ["GeomHex", "geom_hex"]

_SQRT3 = float(np.sqrt(3.0))


class GeomHex(Geom):
    """Hexagons centred on ``(x, y)``, ``width``/``height`` (columns from the
    stat or params; default the data resolution) the bin widths.

    :meth:`setup_data` adds the bounding box (``xmin``/``xmax``/``ymin``/``ymax``)
    so the position scales cover whole hexagons. ``draw_panel`` returns one
    ``VMobject`` per visible row (``.marks``/``.keys`` per row), clipped to the
    panel; opaque outline-less hexagons get a hairline in their fill colour to
    hide anti-aliasing seams (:func:`~._rect.seal_seams`).
    """

    required_aes = ("x", "y")
    default_aes: dict[str, Any] = {
        "color": None,
        "fill": "#7F7F7F",
        "linewidth": 1.0,
        "alpha": 1.0,
        "linetype": "solid",
    }
    extra_params: tuple[str, ...] = ("width", "height")
    default_stat = "bin_hex"
    na_aes: tuple[str, ...] | None = ("x", "y")

    def aesthetics(self) -> frozenset:
        return super().aesthetics() | {"weight"}

    @staticmethod
    def _sizes(data: pd.DataFrame, params) -> tuple[np.ndarray, np.ndarray]:
        params = params or {}
        out = []
        for name, col in (("width", "x"), ("height", "y")):
            v: Any = params.get(name)
            if v is None and name in data.columns:
                v = pd.to_numeric(data[name], errors="coerce").to_numpy(dtype=float)
            if v is None:
                v = resolution(pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float))
            out.append(np.broadcast_to(np.asarray(v, dtype=float), (len(data),)))
        return out[0], out[1]

    def setup_data(self, data: pd.DataFrame, params) -> pd.DataFrame:
        data = data.copy()
        if len(data) == 0:
            for c in ("xmin", "xmax", "ymin", "ymax"):
                data[c] = pd.Series(dtype=float)
            return data
        w, h = self._sizes(data, params)
        x = pd.to_numeric(data["x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(data["y"], errors="coerce").to_numpy(dtype=float)
        data["width"], data["height"] = w, h
        data["xmin"], data["xmax"] = x - w / 2, x + w / 2
        data["ymin"], data["ymax"] = y - h / _SQRT3, y + h / _SQRT3
        return data

    def draw_panel(
        self, data, panel_params, coord, theme: Theme | None = None, width=None, height=None, **p
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        self.check_required_aes(data)
        data = finite_rows(self.use_defaults(data, p), ("x", "y"))
        w, h = self._sizes(data, {"width": width, "height": height})
        hx, hy = hexagon_vertices(data["x"].to_numpy(float), data["y"].to_numpy(float), w, h)
        rect = panel_rect(coord, panel_params)
        out = VGroup()
        keep = []
        for k, row in enumerate(data.to_dict("records")):
            vx, vy = hx[k], hy[k]
            if rect is not None:
                vx, vy = clip_polygon(vx, vy, rect)
            if len(vx) < 3:
                continue
            m = VMobject()
            m.set_points(ring_points(transform_points(coord, vx, vy)))
            m = style_filled(m, row, theme)
            out.add(seal_seams(m) if not isinstance(m, VGroup) else m)
            keep.append(k)
        return attach_marks(out, data.iloc[keep], per="row")

    def draw_key(self, data_row, theme: Theme | None = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        return filled_key(self.key_aes(data_row), theme)


@register
def geom_hex(
    mapping=None,
    data=None,
    *,
    stat="bin_hex",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Hexagonal heatmap of 2-D bin counts of ``aes(x, y)``: ``bins`` (30) or
    ``binwidth`` (one value or an ``(x, y)`` pair); ``fill=after_stat("count")``
    by default. A ``weight`` aesthetic sums weights instead of counting."""
    return _layer(GeomHex(), stat, mapping, data, position, inherit_aes, name, params)
