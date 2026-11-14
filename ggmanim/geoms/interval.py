"""Vertical (or horizontal) intervals: geom_errorbar, geom_errorbarh,
geom_linerange, geom_pointrange and geom_crossbar.

Every geom here draws one mark per (finite, visible) row and is written in the
*vertical* orientation (``x`` and ``ymin``/``ymax``). The horizontal orientation
(``y`` and ``xmin``/``xmax``, ggplot2's ``orientation = "y"``) is the same
drawing with x and y swapped: the data columns are renamed (``x <-> y``,
``xmin <-> ymin`` ...) and the coord is wrapped so that ``transform(x, y)`` calls
the real coord with ``(y, x)``. The orientation is

* ``orientation="x"`` / ``"y"`` when given (``"y"`` = horizontal), else
* the ``flipped_aes`` column written by :meth:`_IntervalGeom.setup_data`, else
* horizontal when ``xmin``/``xmax`` are present and ``ymin``/``ymax`` are not.

``geom_errorbarh`` is ``geom_errorbar`` in the horizontal orientation with
``height`` in place of ``width``.

Widths (errorbar whiskers, crossbar boxes) default to ``0.9 * resolution(x)``
(``0.9`` on a discrete scale), as in ggplot2; ``width=`` (param or column, in
transformed units) overrides. :meth:`setup_data` turns them into
``xmin``/``xmax`` so the x scale trains on the full extent.

Clipping (``clip="on"``) is done in transformed space with
:func:`~ggmanim.geoms._clip.clip_segments` (lines) and
:func:`~ggmanim.geoms._clip.clip_rects` (crossbar boxes); pointrange points are
clipped like ``geom_point`` markers.

:func:`draw_crossbar` is the reusable crossbar drawing (``geom_boxplot`` draws its
box with it).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from manim import DOWN, ORIGIN, UP, Line, Polygon, VGroup, VMobject

from ..registry import register
from ..theme import Theme, theme_default
from ._clip import clip_rects, clip_segments, panel_rect
from ._linetype import apply_linetype
from ._marker_clip import clip_marker, scene_box
from ._markers import make_marker
from .base import (
    KEY_SIZE,
    Geom,
    attach_marks,
    finite_rows,
    make_layer,
    multi_path_mobject,
    num,
    paint_stroke,
    resolve_color,
    transform_points,
)
from .col import resolution

__all__ = [
    "GeomErrorbar",
    "GeomErrorbarh",
    "GeomLinerange",
    "GeomPointrange",
    "GeomCrossbar",
    "geom_errorbar",
    "geom_errorbarh",
    "geom_linerange",
    "geom_pointrange",
    "geom_crossbar",
    "draw_crossbar",
    "flip_data",
]

_SWAP = {
    "x": "y",
    "y": "x",
    "xmin": "ymin",
    "ymin": "xmin",
    "xmax": "ymax",
    "ymax": "xmax",
    "xend": "yend",
    "yend": "xend",
    "xintercept": "yintercept",
    "yintercept": "xintercept",
    "xlower": "ylower",
    "ylower": "xlower",
    "xmiddle": "ymiddle",
    "ymiddle": "xmiddle",
    "xupper": "yupper",
    "yupper": "xupper",
}


def flip_data(data: pd.DataFrame) -> pd.DataFrame:
    """``data`` with every x-like column renamed to its y counterpart and vice
    versa (``x <-> y``, ``xmin <-> ymin`` ...)."""
    return data.rename(columns=_SWAP)


class _FlippedCoord:
    """A coord whose ``transform(x, y)`` is ``coord.transform(y, x)``."""

    def __init__(self, coord: Any):
        self._coord = coord
        self.clip = getattr(coord, "clip", "on")

    def transform(self, x: Any, y: Any) -> np.ndarray:
        return transform_points(self._coord, y, x)


def _flip_panel(coord: Any, panel_params: Any) -> tuple[Any, Any]:
    pp = None
    if panel_params is not None:
        pp = SimpleNamespace(x_range=panel_params.y_range, y_range=panel_params.x_range)
    return _FlippedCoord(coord), pp


def _is_flipped(data: pd.DataFrame, params: Mapping[str, Any] | None, force: bool | None) -> bool:
    if force is not None:
        return force
    orientation = (params or {}).get("orientation")
    if orientation is not None:
        if orientation not in ("x", "y"):
            raise ValueError(f"orientation must be 'x' or 'y', got {orientation!r}")
        return orientation == "y"
    if "flipped_aes" in data.columns and len(data):
        return bool(data["flipped_aes"].iloc[0])
    cols = set(data.columns)
    return {"xmin", "xmax"} <= cols and not {"ymin", "ymax"} <= cols


def _widths(data: pd.DataFrame, params: Mapping[str, Any], discrete: bool) -> np.ndarray:
    """Per-row widths: ``width`` param, else column, else 0.9 * resolution(x)."""
    x = pd.to_numeric(data["x"], errors="coerce").to_numpy(dtype=float)
    width = params.get("width")
    if width is None and "width" in data.columns:
        width = pd.to_numeric(data["width"], errors="coerce").to_numpy(dtype=float)
    if width is None:
        width = 0.9 * (1.0 if discrete else resolution(x))
    return np.broadcast_to(np.asarray(width, dtype=float), x.shape).copy()


def _style(mob: VMobject, row: Mapping[str, Any], fg: Any, linewidth: float) -> VMobject:
    """Stroke ``mob`` with the row's colour/alpha, then dash it (``linetype``)."""
    color: Any = resolve_color(row.get("color"), fg)
    mob.set_fill(opacity=0)
    paint_stroke(mob, color, linewidth, num(row.get("alpha"), 1.0))  # None: blank colour
    return apply_linetype(mob, row.get("linetype"))


def _clipped_segments(coord, rect, x0, y0, x1, y1) -> list[np.ndarray]:
    """The visible parts of the segments ``(x0, y0) -> (x1, y1)`` (transformed
    space) as a list of ``(2, 3)`` scene-point arrays."""
    x0, y0, x1, y1 = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (x0, y0, x1, y1))
    x0, y0, x1, y1 = np.broadcast_arrays(x0, y0, x1, y1)
    if rect is None:
        t0, t1 = np.zeros(len(x0)), np.ones(len(x0))
        vis = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(x1) & np.isfinite(y1)
    else:
        t0, t1, vis = clip_segments(x0, y0, x1, y1, rect)
    out = []
    for i in np.flatnonzero(vis):
        dx, dy = x1[i] - x0[i], y1[i] - y0[i]
        px = [x0[i] + t0[i] * dx, x0[i] + t1[i] * dx]
        py = [y0[i] + t0[i] * dy, y0[i] + t1[i] * dy]
        out.append(transform_points(coord, px, py))
    return out


class _IntervalGeom(Geom):
    """Shared orientation handling; subclasses implement :meth:`_draw_rows` in
    the vertical orientation."""

    #: required columns in the vertical orientation
    vertical_aes: tuple[str, ...] = ("x", "ymin", "ymax")
    required_aes: tuple[str, ...] = ("x", "ymin", "ymax")
    na_aes: tuple[str, ...] | None = ("x", "y", "xmin", "xmax", "ymin", "ymax")
    extra_params: tuple[str, ...] = ("orientation",)
    #: ``True``/``False`` forces the orientation (``geom_errorbarh``)
    force_flipped: bool | None = None
    #: whether :meth:`setup_data` adds ``xmin``/``xmax`` from ``width``
    has_width = False

    def check_required_aes(self, data: pd.DataFrame, flipped: bool = False) -> None:
        need = [_SWAP.get(a, a) for a in self.vertical_aes] if flipped else self.vertical_aes
        missing = [a for a in need if a not in data.columns]
        if missing:
            raise ValueError(
                f"{type(self).__name__} requires the following missing aesthetics: "
                + ", ".join(missing)
            )

    def _params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        return dict(params)

    def setup_data(self, data: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        params = self._params(params or {})
        flipped = _is_flipped(data, params, self.force_flipped)
        data = flip_data(data) if flipped else data.copy()
        if self.has_width and "x" in data.columns:
            discrete = bool(params.get("_y_discrete" if flipped else "_x_discrete"))
            if len(data):
                w = _widths(data, params, discrete)
                x = pd.to_numeric(data["x"], errors="coerce").to_numpy(dtype=float)
                data["xmin"] = x - w / 2
                data["xmax"] = x + w / 2
            else:
                data["xmin"] = pd.Series(dtype=float)
                data["xmax"] = pd.Series(dtype=float)
        data["flipped_aes"] = flipped
        return flip_data(data) if flipped else data

    def draw_panel(
        self, data, panel_params, coord, theme: Theme | None = None, **params
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        params = self._params(params)
        flipped = _is_flipped(data, params, self.force_flipped)
        self.check_required_aes(data, flipped)
        vcols = set((flip_data(data) if flipped else data).columns)
        if self.has_width and not {"xmin", "xmax"} <= vcols:
            data = self.setup_data(data, params)
        data = self.use_defaults(data, params)
        if flipped:
            data = flip_data(data)
            coord, panel_params = _flip_panel(coord, panel_params)
        data = finite_rows(data, [c for c in self.vertical_aes if c != "y"])
        marks, drawn = self._draw_rows(data, panel_params, coord, theme, params)
        return attach_marks(marks, drawn, per="row")

    def _draw_rows(
        self, data: pd.DataFrame, panel_params, coord, theme: Theme, params: Mapping[str, Any]
    ) -> tuple[VGroup, pd.DataFrame]:
        raise NotImplementedError

    def _key_line(self, row, theme) -> VMobject:
        return paint_stroke(
            Line(DOWN * KEY_SIZE / 2, UP * KEY_SIZE / 2),
            resolve_color(row.get("color"), theme.get_foreground()),
            num(row.get("linewidth"), 2.0),
            num(row.get("alpha"), 1.0),
        )


class GeomLinerange(_IntervalGeom):
    """A vertical line from ``ymin`` to ``ymax`` at ``x`` per row (one
    ``VMobject`` mark per visible row)."""

    default_aes = {"color": None, "linewidth": 2.0, "alpha": 1.0, "linetype": "solid"}

    def _draw_rows(self, data, panel_params, coord, theme, params):
        rect = panel_rect(coord, panel_params)
        fg = theme.get_foreground()
        out, drawn = VGroup(), []
        for i, row in enumerate(data.to_dict("records")):
            pieces = _clipped_segments(coord, rect, row["x"], row["ymin"], row["x"], row["ymax"])
            if pieces:
                mob = multi_path_mobject(pieces, None, 2.0, 1.0)
                out.add(_style(mob, row, fg, num(row.get("linewidth"), 2.0)))
                drawn.append(i)
        return out, data.iloc[drawn]

    def draw_key(self, data_row, theme: Theme | None = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        row = self.key_aes(data_row)
        return apply_linetype(self._key_line(row, theme), row.get("linetype"), max_period=KEY_SIZE)


class GeomPointrange(GeomLinerange):
    """A linerange plus a point at ``(x, y)``: one ``VGroup(line, point)`` mark
    per visible row (only the visible parts). The point's radius is
    ``size * fatten`` (``fatten=4``; with the default ``size=0.02`` that is
    0.08 scene units, a bit larger than ``geom_point``'s 0.06, as in ggplot2)."""

    vertical_aes = ("x", "y", "ymin", "ymax")
    required_aes = ("x", "y", "ymin", "ymax")
    default_aes = {
        "color": None,
        "fill": None,
        "size": 0.02,
        "linewidth": 2.0,
        "alpha": 1.0,
        "linetype": "solid",
        "shape": None,
    }
    extra_params: tuple[str, ...] = ("orientation", "fatten")

    def _point(self, point, row, fg, fatten: float) -> VMobject:
        return make_marker(
            row.get("shape"),
            point,
            radius=num(row.get("size"), 0.02) * fatten,
            color=resolve_color(row.get("color"), fg),
            fill=resolve_color(row.get("fill")),
            alpha=num(row.get("alpha"), 1.0),
        )

    def _draw_rows(self, data, panel_params, coord, theme, params):
        rect = panel_rect(coord, panel_params)
        box = scene_box(coord, rect) if rect is not None else None
        fatten = num(params.get("fatten"), 4.0)
        fg = theme.get_foreground()
        out, drawn = VGroup(), []
        for i, row in enumerate(data.to_dict("records")):
            parts = VGroup()
            pieces = _clipped_segments(coord, rect, row["x"], row["ymin"], row["x"], row["ymax"])
            if pieces:
                mob = multi_path_mobject(pieces, None, 2.0, 1.0)
                parts.add(_style(mob, row, fg, num(row.get("linewidth"), 2.0)))
            if np.isfinite(num(row.get("y"), np.nan)):
                p = transform_points(coord, row["x"], row["y"])[0]
                dot: VMobject | None = self._point(p, row, fg, fatten)
                if box is not None and dot is not None:
                    dot = clip_marker(dot, box)
                if dot is not None:
                    parts.add(dot)
            if len(parts):
                out.add(parts)
                drawn.append(i)
        return out, data.iloc[drawn]

    def draw_key(self, data_row, theme: Theme | None = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        row = self.key_aes(data_row)
        line = apply_linetype(self._key_line(row, theme), row.get("linetype"), max_period=KEY_SIZE)
        # the key point is smaller than fatten * size, so it fits the key
        return VGroup(line, self._point(ORIGIN, row, theme.get_foreground(), 2.5))


class GeomErrorbar(_IntervalGeom):
    """Error bars: a vertical line from ``ymin`` to ``ymax`` with whiskers
    ``width`` wide at both ends; one ``VMobject`` (three subpaths: top whisker,
    stem, bottom whisker, only the visible parts) per row."""

    default_aes = {"color": None, "linewidth": 2.0, "alpha": 1.0, "linetype": "solid"}
    extra_params: tuple[str, ...] = ("orientation", "width")
    has_width = True

    def _draw_rows(self, data, panel_params, coord, theme, params):
        rect = panel_rect(coord, panel_params)
        fg = theme.get_foreground()
        out, drawn = VGroup(), []
        for i, row in enumerate(data.to_dict("records")):
            x, lo, hi, a, b = row["x"], row["ymin"], row["ymax"], row["xmin"], row["xmax"]
            pieces = _clipped_segments(
                coord, rect, [a, x, a], [hi, lo, lo], [b, x, b], [hi, hi, lo]
            )
            if pieces:
                mob = multi_path_mobject(pieces, None, 2.0, 1.0)
                out.add(_style(mob, row, fg, num(row.get("linewidth"), 2.0)))
                drawn.append(i)
        return out, data.iloc[drawn]

    def draw_key(self, data_row, theme: Theme | None = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        row = self.key_aes(data_row)
        h, w = KEY_SIZE / 2, KEY_SIZE / 3
        pts = [
            np.array([[-w, h, 0], [w, h, 0]]),
            np.array([[0, -h, 0], [0, h, 0]]),
            np.array([[-w, -h, 0], [w, -h, 0]]),
        ]
        mob = multi_path_mobject(pts, None, 2.0, 1.0)
        mob = _style(mob, row, theme.get_foreground(), num(row.get("linewidth"), 2.0))
        return mob


class GeomErrorbarh(GeomErrorbar):
    """Horizontal error bars: ``y`` with ``xmin``/``xmax``; ``height`` is the
    whisker length (in y units)."""

    vertical_aes = ("x", "ymin", "ymax")  # after flipping: y, xmin, xmax
    extra_params: tuple[str, ...] = ("orientation", "height", "width")
    force_flipped = True

    def _params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        params = dict(params)
        if params.get("height") is not None:
            params["width"] = params["height"]
        return params

    def setup_data(self, data: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        if "height" in data.columns and "width" not in data.columns:
            data = data.rename(columns={"height": "width"})
        return super().setup_data(data, params)


def draw_crossbar(
    rows: pd.DataFrame,
    coord: Any,
    theme: Theme | None = None,
    panel_params: Any = None,
    *,
    fatten: float = 2.5,
    middle: str = "y",
    flipped: bool = False,
) -> tuple[VGroup, pd.DataFrame]:
    """Draw crossbars: a box ``xmin..xmax`` x ``ymin..ymax`` and a line across
    it at ``rows[middle]``, ``fatten`` times thicker than the outline.

    Parameters
    ----------
    rows : DataFrame
        One crossbar per row, in the *vertical* orientation and transformed
        space: ``xmin``, ``xmax``, ``ymin``, ``ymax`` and (optionally finite)
        ``middle``. Style columns ``color``, ``fill``, ``alpha``, ``linewidth``,
        ``linetype`` fall back to :attr:`GeomCrossbar.default_aes`.
    coord, panel_params
        The panel (``coord.transform``) and its params; with ``panel_params``
        and ``coord.clip != "off"`` boxes and middle lines are clipped to the panel.
    theme : Theme, optional
        For the foreground colour (``color=None``).
    fatten : float
        Middle line width = ``linewidth * fatten``.
    middle : str
        Column holding the middle line's position (``NaN``: no line).
    flipped : bool
        Draw horizontally: ``rows`` are still given in the vertical orientation
        (``x`` across the bar) and x/y are swapped when drawing.

    Returns
    -------
    (VGroup, DataFrame)
        One ``VGroup(box, middle_line)`` per visible row (the box is a filled
        ``Polygon``; with a dashed ``linetype`` it is
        ``VGroup(fill, dashes)``), and the rows that were drawn, in order (for
        :func:`~ggmanim.geoms.base.attach_marks`).
    """
    theme = theme if theme is not None else theme_default()
    if flipped:
        coord, panel_params = _flip_panel(coord, panel_params)
    rows = rows.copy()
    for k, v in GeomCrossbar.default_aes.items():
        if k not in rows.columns:
            rows[k] = pd.Series([v] * len(rows), index=rows.index, dtype=object)
    rows = finite_rows(rows, ("xmin", "xmax", "ymin", "ymax"))
    fg = theme.get_foreground()
    rect = panel_rect(coord, panel_params)
    xmin, xmax = rows["xmin"].to_numpy(float), rows["xmax"].to_numpy(float)
    ymin, ymax = rows["ymin"].to_numpy(float), rows["ymax"].to_numpy(float)
    if rect is not None:
        xmin, xmax, ymin, ymax, vis = clip_rects(xmin, xmax, ymin, ymax, rect)
    else:
        vis = np.ones(len(rows), dtype=bool)
    out, drawn = VGroup(), []
    for i, row in enumerate(rows.to_dict("records")):
        if not vis[i]:
            continue
        corners = transform_points(
            coord, [xmin[i], xmax[i], xmax[i], xmin[i]], [ymin[i], ymin[i], ymax[i], ymax[i]]
        )
        lw = num(row.get("linewidth"), 2.0)
        alpha = num(row.get("alpha"), 1.0)
        color: Any = resolve_color(row.get("color"), fg)
        fill = resolve_color(row.get("fill"))
        box: VMobject = Polygon(*corners)
        box.set_fill(fill if fill is not None else color, opacity=alpha if fill is not None else 0)
        paint_stroke(box, color, lw, alpha)
        dashed = apply_linetype(box.copy(), row.get("linetype"))
        if dashed is not box and isinstance(dashed, VGroup):
            box.set_stroke(width=0, opacity=0)
            box = VGroup(box, dashed)
        parts = VGroup(box)
        m = num(row.get(middle), np.nan)
        if np.isfinite(m):
            pieces = _clipped_segments(coord, rect, row["xmin"], m, row["xmax"], m)
            if pieces:
                line = multi_path_mobject(pieces, None, 2.0, 1.0)
                parts.add(_style(line, row, fg, lw * fatten))
        out.add(parts)
        drawn.append(i)
    return out, rows.iloc[drawn]


class GeomCrossbar(_IntervalGeom):
    """Crossbars: a box from ``ymin`` to ``ymax``, ``width`` wide, with a
    ``fatten``-times thicker line at ``y``; one ``VGroup(box, middle)`` per row
    (see :func:`draw_crossbar`). ``fill`` defaults to none."""

    vertical_aes = ("x", "y", "ymin", "ymax")
    required_aes = ("x", "y", "ymin", "ymax")
    default_aes = {
        "color": None,
        "fill": None,
        "linewidth": 2.0,
        "alpha": 1.0,
        "linetype": "solid",
    }
    extra_params: tuple[str, ...] = ("orientation", "width", "fatten")
    has_width = True

    def _draw_rows(self, data, panel_params, coord, theme, params):
        return draw_crossbar(
            data, coord, theme, panel_params, fatten=num(params.get("fatten"), 2.5)
        )

    def draw_key(self, data_row, theme: Theme | None = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        row = pd.DataFrame([{**self.key_aes(data_row), "y": 0.0}])
        s = KEY_SIZE / 2
        row = row.assign(xmin=-s * 0.8, xmax=s * 0.8, ymin=-s, ymax=s)
        key, _ = draw_crossbar(row, _KeyCoord(), theme, None, fatten=1.5)
        return key[0]


class _KeyCoord:
    clip = "off"

    def transform(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])


def _layer(geom, mapping, data, stat, position, inherit_aes, name, params):
    return make_layer(
        geom,
        mapping,
        data,
        stat=stat,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )


@register
def geom_errorbar(
    mapping=None,
    data=None,
    *,
    stat="identity",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Error bars from ``ymin`` to ``ymax`` at ``x`` (or, horizontally, from
    ``xmin`` to ``xmax`` at ``y``; ``orientation="y"`` forces it). ``width=``
    (transformed x units) is the whisker length, default ``0.9 * resolution(x)``."""
    return _layer(GeomErrorbar(), mapping, data, stat, position, inherit_aes, name, params)


@register
def geom_errorbarh(
    mapping=None,
    data=None,
    *,
    stat="identity",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Horizontal error bars from ``xmin`` to ``xmax`` at ``y``; ``height=`` is
    the whisker length (ggplot2 now prefers ``geom_errorbar(orientation="y")``)."""
    return _layer(GeomErrorbarh(), mapping, data, stat, position, inherit_aes, name, params)


@register
def geom_linerange(
    mapping=None,
    data=None,
    *,
    stat="identity",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """A line from ``ymin`` to ``ymax`` at ``x`` (or ``xmin``..``xmax`` at ``y``)."""
    return _layer(GeomLinerange(), mapping, data, stat, position, inherit_aes, name, params)


@register
def geom_pointrange(
    mapping=None,
    data=None,
    *,
    stat="identity",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """A linerange with a point at ``(x, y)``; the point radius is
    ``size * fatten`` (``fatten=4``)."""
    return _layer(GeomPointrange(), mapping, data, stat, position, inherit_aes, name, params)


@register
def geom_crossbar(
    mapping=None,
    data=None,
    *,
    stat="identity",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """A box from ``ymin`` to ``ymax`` with a ``fatten``-times (2.5) thicker
    line at ``y``; ``width=`` as for :func:`geom_errorbar`."""
    return _layer(GeomCrossbar(), mapping, data, stat, position, inherit_aes, name, params)
