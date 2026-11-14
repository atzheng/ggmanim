"""geom_boxplot: box-and-whisker plots from stat_boxplot (ggplot2 semantics).

Each row of the layer data (one per box, from :class:`~ggmanim.stats.boxplot.StatBoxplot`)
is drawn as one mark, ``VGroup(outliers, staples, whiskers, box)`` (only the parts
that exist and are visible):

* the box from ``lower`` to ``upper`` with a median line at ``middle``
  ``fatten`` (2) times thicker than the outline, drawn with
  :func:`~ggmanim.geoms.interval.draw_crossbar` -- or, with ``notch=True``, a
  notched polygon pinched to ``notchwidth`` (0.5) of the box at the median,
  spanning ``notchlower``..``notchupper``;
* whiskers from the hinges to ``ymin``/``ymax``; ``staplewidth`` (0 = none) adds
  staples that wide (relative to the box) at the whisker ends;
* outlier points (``outliers`` list column) at the box's ``x``, styled by the
  ``outlier_*`` params (``color``, ``fill``, ``shape``, ``size``, ``alpha``;
  default: the box's ``color`` and :class:`GeomPoint`'s size), or hidden with
  ``outliers=False`` (which also keeps them out of the y scale).

``setup_data`` turns ``width`` (from the stat: 0.75 on a discrete scale) into
``xmin``/``xmax`` (``varwidth=True`` scales the widths by ``sqrt(n)`` relative to
the largest box) and adds ``ymin_final``/``ymax_final`` (whiskers and outliers)
so the y scale covers the outliers. The default ``position="dodge2"`` places
boxes that share an x side by side (e.g. ``aes(fill=...)``).

Horizontal boxplots (``flipped_aes`` from the stat, ``orientation="y"``) are drawn
in the vertical orientation with x and y swapped. ``fill`` defaults to none
(ggplot2 fills with the paper colour), ``color`` to the theme foreground.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from manim import Polygon, VGroup, VMobject

from ..positions import Position, PositionIdentity, get_position
from ..registry import register
from ..scales.continuous import X_AESTHETICS, Y_AESTHETICS
from ..stats.boxplot import flip_names, resolution, resolve_flipped
from ..theme import Theme, theme_default
from ._clip import clip_polygon, panel_rect
from ._linetype import apply_linetype
from ._marker_clip import clip_marker, scene_box
from ._markers import make_marker
from .base import (
    KEY_SIZE,
    Geom,
    attach_marks,
    make_layer,
    multi_path_mobject,
    num,
    paint_stroke,
    resolve_color,
    transform_points,
)
from .interval import _clipped_segments, _flip_panel, _style, draw_crossbar

__all__ = ["GeomBoxplot", "geom_boxplot"]

#: ``outlier_*`` params (``outlier_colour`` is accepted as ``outlier_color``).
OUTLIER_PARAMS = ("outlier_color", "outlier_fill", "outlier_shape", "outlier_size", "outlier_alpha")


def _is_flipped(data: pd.DataFrame, params: Mapping[str, Any]) -> bool:
    return resolve_flipped(data, None, params.get("orientation"))


class GeomBoxplot(Geom):
    """Boxplots: one ``VGroup`` mark per box row (see the module docstring)."""

    required_aes: tuple[str, ...] = ("x", "lower", "upper", "middle", "ymin", "ymax")
    default_aes = {
        "color": None,
        "fill": None,
        "size": 0.06,
        "alpha": 1.0,
        "shape": None,
        "linewidth": 2.0,
        "linetype": "solid",
    }
    extra_params: tuple[str, ...] = (
        "width",
        "orientation",
        "outliers",
        *OUTLIER_PARAMS,
        "notch",
        "notchwidth",
        "varwidth",
        "staplewidth",
        "fatten",
    )
    default_stat = "boxplot"
    default_position = "dodge2"
    x_aes = (*X_AESTHETICS, "xmin_final", "xmax_final")
    y_aes = (*Y_AESTHETICS, "ymin_final", "ymax_final")
    na_aes: tuple[str, ...] | None = ("x", "y")

    def check_required_aes(self, data: pd.DataFrame, flipped: bool = False) -> None:
        cols = set(flip_names(data).columns if flipped else data.columns)
        missing = [a for a in self.required_aes if a not in cols]
        if missing:
            raise ValueError(
                "GeomBoxplot requires the following missing aesthetics: " + ", ".join(missing)
            )

    def setup_data(self, data: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        params = params or {}
        flipped = _is_flipped(data, params)
        data = flip_names(data) if flipped else data.copy()
        data["flipped_aes"] = flipped
        if not len(data) or "x" not in data.columns:
            return flip_names(data) if flipped else data
        x = pd.to_numeric(data["x"], errors="coerce").to_numpy(dtype=float)
        if "width" in data.columns:
            width = pd.to_numeric(data["width"], errors="coerce").to_numpy(dtype=float)
        elif params.get("width") is not None:
            width = np.full(len(data), float(params["width"]))
        else:
            res = 1.0 if params.get("_y_discrete" if flipped else "_x_discrete") else resolution(x)
            width = np.full(len(data), 0.9 * res)
        if params.get("outliers") is False and "outliers" in data.columns:
            data = data.drop(columns=["outliers"])
        if "outliers" in data.columns and {"ymin", "ymax"} <= set(data.columns):
            lo = [float(np.min(o)) if len(o) else np.inf for o in data["outliers"]]
            hi = [float(np.max(o)) if len(o) else -np.inf for o in data["outliers"]]
            data["ymin_final"] = np.minimum(lo, data["ymin"].to_numpy(dtype=float))
            data["ymax_final"] = np.maximum(hi, data["ymax"].to_numpy(dtype=float))
        rel = np.ones(len(data))
        if params.get("varwidth") and "relvarwidth" in data.columns:
            rv = pd.to_numeric(data["relvarwidth"], errors="coerce").to_numpy(dtype=float)
            rel = rv / np.nanmax(rv)
        data["xmin"] = x - rel * width / 2
        data["xmax"] = x + rel * width / 2
        data = data.drop(columns=["width", "relvarwidth"], errors="ignore")
        return flip_names(data) if flipped else data

    def draw_panel(
        self, data, panel_params, coord, theme: Theme | None = None, **params
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        if not len(data):
            return attach_marks(VGroup(), data, per="row")
        flipped = _is_flipped(data, params)
        self.check_required_aes(data, flipped)
        vdata = flip_names(data) if flipped else data
        if not {"xmin", "xmax"} <= set(vdata.columns):
            data = self.setup_data(data, params)
        data = self.use_defaults(data, params)
        if flipped:
            data = flip_names(data)
            coord, panel_params = _flip_panel(coord, panel_params)
        out, drawn = VGroup(), []
        for i, row in enumerate(data.to_dict("records")):
            mark = self._draw_box(row, coord, panel_params, theme, params)
            if mark is not None:
                out.add(mark)
                drawn.append(i)
        return attach_marks(out, data.iloc[drawn], per="row")

    # -- parts ------------------------------------------------------------------
    def _draw_box(self, row, coord, panel_params, theme, params) -> VGroup | None:
        vals = [num(row.get(c), np.nan) for c in ("x", "xmin", "xmax", "lower", "upper")]
        if not np.all(np.isfinite(vals)):
            return None
        x, xmin, xmax, lower, upper = vals
        fg = theme.get_foreground()
        rect = panel_rect(coord, panel_params)
        lw = num(row.get("linewidth"), 2.0)
        parts = VGroup()
        outliers = self._outliers(row, coord, rect, theme, params)
        if outliers is not None:
            parts.add(outliers)
        staple = num(params.get("staplewidth"), 0.0)
        ymin, ymax = num(row.get("ymin"), lower), num(row.get("ymax"), upper)
        if staple:
            a, b = x + (xmin - x) * staple, x + (xmax - x) * staple
            pieces = _clipped_segments(coord, rect, [a, a], [ymax, ymin], [b, b], [ymax, ymin])
            if pieces:
                parts.add(_style(multi_path_mobject(pieces, None, lw, 1.0), row, fg, lw))
        pieces = _clipped_segments(coord, rect, [x, x], [upper, lower], [x, x], [ymax, ymin])
        if pieces:
            parts.add(_style(multi_path_mobject(pieces, None, lw, 1.0), row, fg, lw))
        fatten = num(params.get("fatten"), 2.0)
        if params.get("notch"):
            box = self._notched(row, coord, rect, theme, num(params.get("notchwidth"), 0.5), fatten)
        else:
            rows = pd.DataFrame([{**row, "ymin": lower, "ymax": upper}])
            boxes, _ = draw_crossbar(
                rows, coord, theme, panel_params, fatten=fatten, middle="middle"
            )
            box = boxes[0] if len(boxes) else None
        if box is not None:
            parts.add(box)
        return parts if len(parts) else None

    def _outliers(self, row, coord, rect, theme, params) -> VGroup | None:
        values = row.get("outliers")
        if values is None or params.get("outliers") is False:
            return None
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return None
        box = scene_box(coord, rect) if rect is not None else None
        color = params.get("outlier_color")
        color = resolve_color(
            color if color is not None else row.get("color"), theme.get_foreground()
        )
        shape = params.get("outlier_shape")
        size = params.get("outlier_size")
        alpha = params.get("outlier_alpha")
        pts = transform_points(coord, np.full(len(values), row["x"]), values)
        out = VGroup()
        for p in pts:
            dot: VMobject | None = make_marker(
                shape if shape is not None else row.get("shape"),
                p,
                radius=num(size if size is not None else row.get("size"), 0.06),
                color=color,
                fill=resolve_color(params.get("outlier_fill")),
                alpha=num(alpha if alpha is not None else row.get("alpha"), 1.0),
            )
            if box is not None and dot is not None:
                dot = clip_marker(dot, box)
            if dot is not None:
                out.add(dot)
        return out if len(out) else None

    def _notched(self, row, coord, rect, theme, notchwidth: float, fatten: float):
        xmin, xmax = float(row["xmin"]), float(row["xmax"])
        lower, upper, mid = float(row["lower"]), float(row["upper"]), float(row["middle"])
        nlo, nhi = num(row.get("notchlower"), mid), num(row.get("notchupper"), mid)
        indent = (1 - notchwidth) * (xmax - xmin) / 2
        px = [xmin, xmin, xmin + indent, xmin, xmin, xmax, xmax, xmax - indent, xmax, xmax]
        py = [upper, nhi, mid, nlo, lower, lower, nlo, mid, nhi, upper]
        px_, py_ = np.asarray(px), np.asarray(py)
        if rect is not None:
            px_, py_ = clip_polygon(px_, py_, rect)
        if len(px_) < 3:
            return None
        fg = theme.get_foreground()
        color: Any = resolve_color(row.get("color"), fg)
        fill = resolve_color(row.get("fill"))
        alpha = num(row.get("alpha"), 1.0)
        lw = num(row.get("linewidth"), 2.0)
        poly: VMobject = Polygon(*transform_points(coord, px_, py_))
        poly.set_fill(fill if fill is not None else color, opacity=alpha if fill is not None else 0)
        paint_stroke(poly, color, lw, alpha)
        dashed = apply_linetype(poly.copy(), row.get("linetype"))
        if dashed is not poly and isinstance(dashed, VGroup):
            poly.set_stroke(width=0, opacity=0)
            poly = VGroup(poly, dashed)
        parts = VGroup(poly)
        pieces = _clipped_segments(coord, rect, xmin + indent, mid, xmax - indent, mid)
        if pieces:
            parts.add(_style(multi_path_mobject(pieces, None, lw, 1.0), row, fg, lw * fatten))
        return parts

    def draw_key(self, data_row, theme: Theme | None = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        row = self.key_aes(data_row)
        s = KEY_SIZE / 2
        rows = pd.DataFrame(
            [
                {
                    **row,
                    "xmin": -0.75 * s,
                    "xmax": 0.75 * s,
                    "ymin": -0.5 * s,
                    "ymax": 0.5 * s,
                    "middle": 0.0,
                }
            ]
        )
        coord = _SceneCoord()
        box, _ = draw_crossbar(rows, coord, theme, None, fatten=1.5, middle="middle")
        fg = theme.get_foreground()
        lw = num(row.get("linewidth"), 2.0)
        pieces = _clipped_segments(coord, None, [0, 0], [0.5 * s, -0.5 * s], [0, 0], [s, -s])
        whiskers = _style(multi_path_mobject(pieces, None, lw, 1.0), row, fg, lw)
        return VGroup(whiskers, box[0])


class _SceneCoord:
    """Identity coord (legend keys are drawn in scene units)."""

    clip = "off"

    def transform(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])


class OrientedPosition(Position):
    """Runs ``position`` on horizontal layers (``flipped_aes`` true) with x and y
    swapped, as ggplot2's positions do, so ``position_dodge``/``position_dodge2``
    place horizontal boxplots and violins side by side along y. Vertical layers
    are passed through unchanged. (Not registered: no ``name``.)"""

    def __init__(self, position: Any) -> None:
        super().__init__(position=get_position(position))

    @property
    def inner(self) -> Position:
        return self.params["position"]

    def __repr__(self) -> str:
        return f"{self.inner!r}"

    @staticmethod
    def _flipped(data: pd.DataFrame) -> bool:
        return "flipped_aes" in data.columns and len(data) > 0 and bool(data["flipped_aes"].all())

    def setup_params(self, data):
        flipped = self._flipped(data)
        inner = self.inner.setup_params(flip_names(data) if flipped else data)
        return {"inner": inner, "flipped": flipped}

    def setup_data(self, data, params):
        if not params["flipped"]:
            return self.inner.setup_data(data, params["inner"])
        return flip_names(self.inner.setup_data(flip_names(data), params["inner"]))

    def compute_layer(self, data, params, scales=None):
        if not params["flipped"]:
            return self.inner.compute_layer(data, params["inner"], scales)
        swapped = {"x": scales.get("y"), "y": scales.get("x")} if scales else scales
        return flip_names(self.inner.compute_layer(flip_names(data), params["inner"], swapped))


def oriented(position: Any) -> Any:
    """``position`` wrapped in :class:`OrientedPosition` (identity as is)."""
    pos = get_position(position)
    if isinstance(pos, (PositionIdentity, OrientedPosition)):
        return pos
    return OrientedPosition(pos)


def _standardise_outlier_params(params: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in params.items():
        if k.startswith("outlier_") and k[8:] in ("colour", "col"):
            k = "outlier_color"
        out[k] = v
    return out


@register
def geom_boxplot(
    mapping=None,
    data=None,
    *,
    stat="boxplot",
    position="dodge2",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Box-and-whisker plots (ggplot2's ``geom_boxplot``); see
    :mod:`ggmanim.geoms.boxplot`.

    Stat params: ``coef`` (1.5, whisker reach in IQRs), ``width``. Geom params:
    ``outliers`` (``False`` hides them), ``outlier_color``/``outlier_fill``/
    ``outlier_shape``/``outlier_size``/``outlier_alpha``, ``notch``,
    ``notchwidth``, ``varwidth``, ``staplewidth``, ``fatten`` (median line, 2),
    ``orientation``. Default position ``"dodge2"``."""
    return make_layer(
        GeomBoxplot(),
        mapping,
        data,
        stat=stat,
        position=oriented(position),
        inherit_aes=inherit_aes,
        name=name,
        params=_standardise_outlier_params(params),
    )
