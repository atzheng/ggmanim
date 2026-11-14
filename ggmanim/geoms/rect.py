"""geom_rect, geom_tile, geom_raster: axis-aligned rectangles.

* :func:`geom_rect` -- rectangles given by ``xmin``/``xmax``/``ymin``/``ymax``.
* :func:`geom_tile` -- rectangles centred on ``(x, y)`` with ``width``/``height``
  (default: the data resolution, 1 on a discrete axis); the heatmap primitive
  (``aes(fill=value)`` uses the continuous fill gradient).
* :func:`geom_raster` -- a fast tile variant for regular grids: one ``VMobject``
  per distinct fill colour, every tile of that colour a subpath.

All three go through ``coord.transform`` (so they work under scale transforms and
``coord_flip``) and are clamped to the panel when clipping is on.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from manim import VGroup, VMobject

from ..coords import is_linear
from ..registry import register
from ..theme import Theme, theme_default
from ._rect import clipped_rects, draw_rects, filled_key, ring_points, seal_seams
from .base import Geom, attach_marks, finite_rows, make_layer, num, resolve_color
from .col import resolution

__all__ = ["GeomRect", "GeomTile", "GeomRaster", "geom_rect", "geom_tile", "geom_raster"]

_RECT_COLS = ("xmin", "xmax", "ymin", "ymax")


class GeomRect(Geom):
    """Rectangles. ``draw_panel`` returns a ``VGroup`` with one ``VMobject`` per
    visible row, in data order (``.marks``/``.keys``: one per drawn row); a
    dashed outline (``linetype``) makes a row's mark ``VGroup(fill, dashes)``.

    ``color`` (outline) defaults to ``None`` = no outline; ``alpha`` applies to the
    fill only (as in ggplot2).
    """

    required_aes: tuple[str, ...] = _RECT_COLS
    default_aes: dict[str, Any] = {
        "color": None,
        "fill": "#595959",
        "linewidth": 1.0,
        "alpha": 1.0,
        "linetype": "solid",
    }
    na_aes: tuple[str, ...] | None = _RECT_COLS
    #: Hide anti-aliasing seams between abutting rectangles (see ``seal_seams``).
    seal = False

    def draw_panel(
        self, data, panel_params, coord, theme: Theme | None = None, **params
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        if not set(_RECT_COLS) <= set(data.columns):
            data = self.setup_data(data, params)
        self.check_required_aes(data)
        data = finite_rows(self.use_defaults(data, params), _RECT_COLS)
        out, drawn = draw_rects(data, panel_params, coord, theme, seal=self.seal)
        return attach_marks(out, drawn, per="row")

    def draw_key(self, data_row, theme: Theme | None = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        return filled_key(self.key_aes(data_row), theme)


def _extent(data: pd.DataFrame, params, name: str, pos: np.ndarray, discrete: bool) -> np.ndarray:
    """``width``/``height`` of each tile: the param, else a mapped column, else the
    resolution of ``pos`` (1 on a discrete axis)."""
    value: Any = params.get(name) if params else None
    if value is None and name in data.columns:
        value = pd.to_numeric(data[name], errors="coerce").to_numpy(dtype=float)
    if value is None:
        value = 1.0 if discrete else resolution(pos)
    return np.broadcast_to(np.asarray(value, dtype=float), pos.shape)


class GeomTile(GeomRect):
    """Tiles centred on ``(x, y)``.

    :meth:`setup_data` (transformed space) adds ``xmin = x - width/2``,
    ``xmax = x + width/2``, ``ymin = y - height/2``, ``ymax = y + height/2``.
    ``width``/``height`` (params or mapped columns, transformed units) default to
    ``resolution(x)``/``resolution(y)``, or 1 on a discrete axis. Drawn like
    :class:`GeomRect` (one mark per drawn row); opaque tiles without an outline
    get a hairline stroke in their fill colour so no anti-aliasing seams show
    between neighbours (:func:`~._rect.seal_seams`).
    """

    required_aes = ("x", "y")
    extra_params: tuple[str, ...] = ("width", "height")
    seal = True

    def setup_data(self, data: pd.DataFrame, params) -> pd.DataFrame:
        data = data.copy()
        if len(data) == 0:
            for c in _RECT_COLS:
                data[c] = pd.Series(dtype=float)
            return data
        params = params or {}
        x = pd.to_numeric(data["x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(data["y"], errors="coerce").to_numpy(dtype=float)
        w = _extent(data, params, "width", x, bool(params.get("_x_discrete")))
        h = _extent(data, params, "height", y, bool(params.get("_y_discrete")))
        data["xmin"], data["xmax"] = x - w / 2, x + w / 2
        data["ymin"], data["ymax"] = y - h / 2, y + h / 2
        return data

    def draw_panel(
        self, data, panel_params, coord, theme: Theme | None = None, width=None, height=None, **p
    ) -> VMobject:
        if not set(_RECT_COLS) <= set(data.columns):
            data = self.setup_data(data, {"width": width, "height": height})
        return super().draw_panel(data, panel_params, coord, theme, **p)


def _regular(values: np.ndarray) -> bool:
    """Are the distinct finite ``values`` on one regular lattice (every gap an
    integer multiple of the smallest)?"""
    u = np.unique(values[np.isfinite(values)])
    if len(u) < 3:
        return True
    d = np.diff(u)
    step = d.min()
    if step <= 0:
        return False
    k = d / step
    return bool(np.allclose(k, np.round(k), rtol=0, atol=1e-6))


class GeomRaster(GeomTile):
    """Fast tiles for a regular grid (``hjust``/``vjust`` place the cell
    relative to ``(x, y)``, default centred).

    When the ``(x, y)`` positions lie on a regular lattice without duplicate cells,
    ``draw_panel`` returns a ``VGroup`` with **one** ``VMobject`` per distinct
    ``(fill, alpha)``, each tile of that colour a closed subpath, in order of first
    appearance; ``.marks`` are those mobjects and ``.keys`` their fill colours (hex
    strings). Otherwise it falls back to :class:`GeomTile` drawing (one mark per
    row). Tile sizes are always the resolutions (no ``width``/``height``); there
    is no outline (``color``/``linetype`` are not raster aesthetics).
    """

    default_aes: dict[str, Any] = {"fill": "#595959", "alpha": 1.0}
    extra_params: tuple[str, ...] = ("hjust", "vjust")

    def setup_data(self, data: pd.DataFrame, params) -> pd.DataFrame:
        params = dict(params or {})
        data = super().setup_data(data, {**params, "width": None, "height": None})
        if len(data):
            hjust = num(params.get("hjust"), 0.5)
            vjust = num(params.get("vjust"), 0.5)
            w = data["xmax"] - data["xmin"]
            h = data["ymax"] - data["ymin"]
            x = pd.to_numeric(data["x"], errors="coerce")
            y = pd.to_numeric(data["y"], errors="coerce")
            data["xmin"], data["xmax"] = x - w * (1 - hjust), x + w * hjust
            data["ymin"], data["ymax"] = y - h * (1 - vjust), y + h * vjust
        return data

    def draw_panel(
        self,
        data,
        panel_params,
        coord,
        theme: Theme | None = None,
        width=None,
        height=None,
        hjust=None,
        vjust=None,
        **p,
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        if not set(_RECT_COLS) <= set(data.columns):
            data = self.setup_data(data, {"hjust": hjust, "vjust": vjust})
        self.check_required_aes(data)
        data = finite_rows(self.use_defaults(data, p), _RECT_COLS)
        x = pd.to_numeric(data["x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(data["y"], errors="coerce").to_numpy(dtype=float)
        cells = pd.DataFrame({"x": x, "y": y})
        if not (_regular(x) and _regular(y) and is_linear(coord)) or cells.duplicated().any():
            out, drawn = draw_rects(data, panel_params, coord, theme, seal=True)
            return attach_marks(out, drawn, per="row")
        data, corners = clipped_rects(data, coord, panel_params)
        # tiles clamped to zero size on the panel border would only add degenerate subpaths
        diag = np.abs(corners[:, 2, :2] - corners[:, 0, :2])
        keep = (diag > 1e-12).all(axis=1)
        data, corners = data[keep], corners[keep]
        fg = theme.get_foreground()
        fills = [resolve_color(v, fg) for v in data["fill"]]
        alphas = [num(v, 1.0) for v in data["alpha"]]
        batches: dict[tuple[str, float], list[int]] = {}
        for i, (f, a) in enumerate(zip(fills, alphas)):
            batches.setdefault((f.to_hex() if f is not None else "", a), []).append(i)
        out = VGroup()
        for (_hexcol, alpha), idx in batches.items():
            m = VMobject()
            m.set_points(np.concatenate([ring_points(corners[i]) for i in idx], axis=0))
            m.set_fill(fills[idx[0]], opacity=alpha)
            m.set_stroke(width=0, opacity=0)
            out.add(seal_seams(m))
        keys = pd.DataFrame({"key": [k[0] for k in batches]})
        return attach_marks(out, keys, per="row", batched=True)


@register
def geom_rect(
    mapping=None,
    data=None,
    *,
    stat="identity",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Rectangles from ``aes(xmin=, xmax=, ymin=, ymax=)``. Constant ``fill``,
    ``color`` (outline), ``alpha`` (fill only), ``linewidth``, ``linetype`` go in
    ``params``."""
    return make_layer(
        GeomRect(),
        mapping,
        data,
        stat=stat,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )


@register
def geom_tile(
    mapping=None,
    data=None,
    *,
    stat="identity",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Tiles centred on ``aes(x, y)``: heatmaps with ``aes(fill=value)``.

    ``width=``/``height=`` (params or mapped) are in transformed units and default
    to the data resolution (1 on a discrete axis)."""
    return make_layer(
        GeomTile(),
        mapping,
        data,
        stat=stat,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )


@register
def geom_raster(
    mapping=None,
    data=None,
    *,
    stat="identity",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Like :func:`geom_tile` for a regular grid, drawn as one mobject per fill
    colour (much faster for large heatmaps). ``hjust``/``vjust`` (default 0.5)
    place each cell relative to its ``(x, y)``."""
    return make_layer(
        GeomRaster(),
        mapping,
        data,
        stat=stat,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
