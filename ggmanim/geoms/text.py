"""geom_text / geom_label: text at data positions (label on a box for geom_label).

Text ``size`` follows ggplot2's millimetre scale relative to the theme:
``size=3.88`` (the default) is drawn at ``theme.axis_text_size`` and other sizes
scale linearly (``font_size = theme.axis_text_size * size / 3.88``). See
:mod:`._labels` for justification (``hjust``/``vjust``/``angle``) semantics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from manim import Rectangle, RoundedRectangle, VGroup, VMobject

from ..registry import register
from ..theme import Theme, foreground_for_background, theme_default
from ._labels import (
    bbox,
    boxes_overlap,
    font_size_for,
    place_text,
    resolve_just,
    text_mobject,
)
from .base import (
    KEY_SIZE,
    Geom,
    attach_marks,
    finite_rows,
    make_layer,
    num,
    resolve_color,
    transform_points,
)

__all__ = ["GeomText", "GeomLabel", "geom_text", "geom_label"]


class GeomText(Geom):
    """Text labels: ``draw_panel`` returns a ``VGroup`` with one text mobject per
    drawn row, in data order (``.marks``/``.keys`` per row).

    Each label's ``(hjust, vjust)`` point (of its unrotated bounding box) sits at
    ``coord.transform(x, y)``; the text is then rotated by ``angle`` degrees about
    that point. Rows with a missing/empty ``label`` are not drawn.

    Aesthetics: ``x``, ``y``, ``label`` (required); ``color`` (``None`` = theme
    foreground), ``size`` (ggplot2 mm, see module docstring), ``alpha``,
    ``angle`` (degrees), ``hjust``/``vjust`` (``0``..``1`` or ``"left"``,
    ``"right"``, ``"top"``, ``"bottom"``, ``"center"``, ``"inward"``,
    ``"outward"``), ``family`` (Pango font), ``fontface`` (``"plain"``,
    ``"bold"``, ``"italic"``, ``"bold.italic"``).

    Params: ``parse`` (LaTeX math via ``MathTex`` when the theme's
    ``text_engine`` is ``"tex"``, else plain text with a warning),
    ``nudge_x``/``nudge_y`` (offsets in the scale's transformed units -- data
    units on linear scales -- applied by :meth:`setup_data`, so nudged labels take
    part in the axis limits), ``check_overlap`` (drop a label whose bounding box
    overlaps one already drawn, greedily in data order).
    """

    required_aes = ("x", "y", "label")
    default_aes: dict[str, Any] = {
        "color": None,
        "size": 3.88,
        "alpha": 1.0,
        "angle": 0.0,
        "hjust": 0.5,
        "vjust": 0.5,
        "family": None,
        "fontface": "plain",
    }
    extra_params: tuple[str, ...] = ("parse", "nudge_x", "nudge_y", "check_overlap")
    na_aes: tuple[str, ...] | None = ("x", "y")

    # -- pipeline hooks -----------------------------------------------------------
    def setup_data(self, data: pd.DataFrame, params) -> pd.DataFrame:
        """Apply ``nudge_x``/``nudge_y`` (transformed units) to ``x``/``y``."""
        params = params or {}
        nx, ny = num(params.get("nudge_x"), 0.0), num(params.get("nudge_y"), 0.0)
        if not (nx or ny) or len(data) == 0:
            return data
        data = data.copy()
        for col, d in (("x", nx), ("y", ny)):
            if d and col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce") + d
        return data

    # -- drawing ------------------------------------------------------------------
    def _draw_opts(self, params: dict) -> dict:
        """Pop the geom-specific drawing params from ``params`` (for :meth:`_mobject`)."""
        return {}

    def _mobject(self, row: dict, theme: Theme, parse: bool, **opts) -> VMobject | None:
        """The (unplaced) mobject for one row; ``None`` if the label is empty."""
        return text_mobject(
            theme,
            row.get("label"),
            font_size=font_size_for(theme, row.get("size")),
            color=resolve_color(row.get("color"), theme.get_foreground()),
            alpha=num(row.get("alpha"), 1.0),
            family=row.get("family"),
            fontface=row.get("fontface"),
            parse=parse,
            hjust=resolve_just(row.get("hjust")),
        )  # type: ignore[return-value]

    def draw_panel(
        self,
        data,
        panel_params,
        coord,
        theme: Theme | None = None,
        parse: bool = False,
        check_overlap: bool = False,
        nudge_x=None,
        nudge_y=None,
        **params,
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        opts = self._draw_opts(params)
        data = self.use_defaults(data, params)
        self.check_required_aes(data)
        data = finite_rows(data, ("x", "y"))
        pts = transform_points(coord, data["x"].to_numpy(float), data["y"].to_numpy(float))
        rel = _relative_positions(coord, panel_params, pts)
        out = VGroup()
        drawn: list[int] = []
        kept: list[np.ndarray] = []
        for i, (p, row) in enumerate(zip(pts, data.to_dict("records"))):
            mob = self._mobject(row, theme, parse, **opts)
            if mob is None:
                continue
            hjust = resolve_just(row.get("hjust"), rel[i, 0] if rel is not None else None)
            vjust = resolve_just(
                row.get("vjust"), rel[i, 1] if rel is not None else None, vertical=True
            )
            place_text(mob, p, hjust, vjust, num(row.get("angle"), 0.0))
            if check_overlap:
                box = bbox(mob)
                if any(boxes_overlap(box, k) for k in kept):
                    continue
                kept.append(box)
            out.add(mob)
            drawn.append(i)
        return attach_marks(out, data.iloc[drawn], per="row")

    def draw_key(self, data_row, theme: Theme | None = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        row = {**self.key_aes(data_row), "label": "a", "angle": 0.0}
        mob = self._mobject(row, theme, parse=False)
        assert mob is not None
        mob.scale_to_fit_height(0.6 * KEY_SIZE).move_to([0, 0, 0])
        return mob  # type: ignore[return-value]


class GeomLabel(GeomText):
    """Like :class:`GeomText`, each label on a (rounded) background box.

    The box is ``label_padding`` scene units larger than the text on every side,
    has corner radius ``label_r`` (scene units; ``0`` = square corners), is
    filled with ``fill`` and outlined in the text colour with stroke width
    ``label_size`` (``0`` = no outline). Unset colours contrast with each other
    (:func:`~ggmanim.theme.foreground_for_background`): with neither set the text
    is the theme foreground on a contrasting box; with only ``fill`` the text is
    black or white to suit it; with only ``color`` the box contrasts with the
    text. Justification and
    ``angle`` apply to the box. ``alpha`` applies to the whole label. Each mark
    is ``VGroup(box, text)``.
    """

    default_aes: dict[str, Any] = {**GeomText.default_aes, "fill": None}
    extra_params: tuple[str, ...] = GeomText.extra_params + (
        "label_padding",
        "label_r",
        "label_size",
    )

    def _draw_opts(self, params: dict) -> dict:
        return {
            "pad": num(params.pop("label_padding", None), 0.08),
            "r": num(params.pop("label_r", None), 0.05),
            "stroke": num(params.pop("label_size", None), 1.0),
        }

    def _mobject(
        self,
        row: dict,
        theme: Theme,
        parse: bool,
        pad: float = 0.08,
        r: float = 0.05,
        stroke: float = 1.0,
        **opts,
    ) -> VMobject | None:
        color, fill = resolve_color(row.get("color")), resolve_color(row.get("fill"))
        if color is None:
            color = theme.get_foreground() if fill is None else foreground_for_background(fill)
        if fill is None:
            fill = foreground_for_background(color)
        text = super()._mobject({**row, "alpha": 1.0, "color": color}, theme, parse)
        if text is None:
            return None
        w, h = text.width + 2 * pad, text.height + 2 * pad
        r = min(max(r, 0.0), w / 2, h / 2)
        box = (
            RoundedRectangle(width=w, height=h, corner_radius=r)
            if r > 0
            else Rectangle(width=w, height=h)
        )
        box.move_to(text.get_center())
        alpha = num(row.get("alpha"), 1.0)
        box.set_fill(fill, opacity=alpha)
        if stroke > 0:
            box.set_stroke(color=color, width=stroke, opacity=alpha)
        else:
            box.set_stroke(width=0, opacity=0)
        if alpha < 1.0:
            text.set_opacity(alpha)
        return VGroup(box, text)


def _relative_positions(coord, panel_params, pts: np.ndarray) -> np.ndarray | None:
    """Each scene point's ``(x, y)`` position relative to the panel (``0``..``1``),
    for ``"inward"``/``"outward"`` justification; ``None`` without a panel."""
    if panel_params is None or not len(pts):
        return None
    (x0, x1), (y0, y1) = panel_params.x_range, panel_params.y_range
    corners = transform_points(coord, [x0, x1], [y0, y1])
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return ((pts - lo) / span)[:, :2]


def _text_layer(geom, mapping, data, stat, position, inherit_aes, name, params):
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
def geom_text(
    mapping=None,
    data=None,
    *,
    stat="identity",
    position="identity",
    parse: bool = False,
    nudge_x: float = 0.0,
    nudge_y: float = 0.0,
    check_overlap: bool = False,
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Text at ``(x, y)``: map ``aes(label=...)`` (or pass a constant ``label=``).

    ``size=3.88`` (ggplot2 mm) is ``theme.axis_text_size``; ``hjust``/``vjust``
    in ``[0, 1]`` or ``"inward"``/``"outward"``; ``angle`` in degrees;
    ``nudge_x``/``nudge_y`` shift the labels in (transformed) data units;
    ``check_overlap=True`` drops labels overlapping earlier ones;
    ``parse=True`` renders LaTeX math (needs ``theme(text_engine="tex")``).
    Counts on bars: ``geom_text(aes(label=after_stat("count")), stat="count")``.
    See :class:`GeomText`.
    """
    params.update(parse=parse, nudge_x=nudge_x, nudge_y=nudge_y, check_overlap=check_overlap)
    return _text_layer(GeomText(), mapping, data, stat, position, inherit_aes, name, params)


@register
def geom_label(
    mapping=None,
    data=None,
    *,
    stat="identity",
    position="identity",
    parse: bool = False,
    nudge_x: float = 0.0,
    nudge_y: float = 0.0,
    label_padding: float = 0.08,
    label_r: float = 0.05,
    label_size: float = 1.0,
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Like :func:`geom_text` with a background box: ``label_padding`` and
    ``label_r`` (corner radius) in scene units, ``label_size`` the outline's
    stroke width, ``fill`` the box colour (default white). See :class:`GeomLabel`."""
    params.update(
        parse=parse,
        nudge_x=nudge_x,
        nudge_y=nudge_y,
        label_padding=label_padding,
        label_r=label_r,
        label_size=label_size,
    )
    return _text_layer(GeomLabel(), mapping, data, stat, position, inherit_aes, name, params)
