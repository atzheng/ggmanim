"""geom_vline / geom_hline: reference lines spanning the whole panel.

Under a non-linear coord (``coord_trans``, ``coord_polar``) each line is
munched (`_segments.munched_segment()`) so it bends with the coord: under
``coord_polar`` a reference line on the radius aesthetic becomes a circle and
one on the angle aesthetic a ray from the centre.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from manim import DOWN, LEFT, RIGHT, UP, Line, VGroup, VMobject

from ..aes import aes
from ..registry import register
from ..theme import Theme, theme_default
from ._clip import inside
from ._labels import direct_label
from ._linetype import apply_linetype
from ._segments import munched_segment
from .base import (
    KEY_SIZE,
    Geom,
    assemble,
    attach_marks,
    finite_rows,
    group_label,
    make_layer,
    multi_path_mobject,
    num,
    paint_stroke,
    resolve_color,
    transform_points,
)

__all__ = ["GeomVline", "GeomHline", "geom_vline", "geom_hline"]


class _RefLine(Geom):
    """Shared implementation; ``vertical`` selects vline/hline.

    ``draw_panel`` returns ``VGroup(*lines, *labels)`` (``.marks`` / ``.labels``):
    one ``Line`` per row spanning the panel's full y (vline) or x (hline) range
    from ``panel_params`` (under a non-linear coord a munched path instead: a
    circle or a ray under ``coord_polar``). ``label=`` (str/Mobject for every line, list in row
    order or dict row-index -> label) is drawn in the line's colour: rotated along
    the top-left of a vline, above the right end of an hline.
    """

    default_aes = {"color": None, "linewidth": 2.0, "alpha": 1.0, "linetype": "solid"}
    extra_params: tuple[str, ...] = ("label",)
    #: ``label=`` dicts are keyed by row position, not group (see ``build.draw_layers``)
    label_by = "row"
    vertical = True

    @property
    def _col(self):
        return "xintercept" if self.vertical else "yintercept"

    def draw_panel(
        self, data, panel_params, coord, theme: Theme | None = None, label=None, **params
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        self.check_required_aes(data)
        data = finite_rows(self.use_defaults(data, params), (self._col,))
        fg = theme.get_foreground()
        v = data[self._col].to_numpy(float)
        if panel_params is not None and getattr(coord, "clip", "on") != "off":
            lo, hi = sorted(panel_params.x_range if self.vertical else panel_params.y_range)
            vis = inside(v, 0.0, (lo, hi, -1.0, 1.0))  # drop lines outside the panel
            v, data = v[vis], data[vis]
        lo, hi = (
            float(r) for r in (panel_params.y_range if self.vertical else panel_params.x_range)
        )
        ones = np.ones_like(v)
        x0, y0, x1, y1 = (
            (v, lo * ones, v, hi * ones) if self.vertical else (lo * ones, v, hi * ones, v)
        )
        munch = not getattr(coord, "is_linear", True)
        # coord_radar is linear, so a line along theta would collapse onto itself
        # (both ends at the same angle): draw a closed polygon through the spokes
        spokes = self._radar_spokes(coord, panel_params)
        a, b = transform_points(coord, x0, y0), transform_points(coord, x1, y1)
        lines, labels, drawn = VGroup(), VGroup(), []
        for i, row in enumerate(data.to_dict("records")):
            color = resolve_color(row.get("color"), fg)
            alpha = num(row.get("alpha"), 1.0)
            width = num(row.get("linewidth"), 2.0)
            if spokes is not None:  # polygon through the spokes at this radius
                ring = transform_points(coord, *self._ring(spokes, v[i]))
                line = multi_path_mobject([ring], color, width, alpha)
                q = ring[int(np.argmax(ring[:, 1]))]
            elif munch:  # bends with the coord: circles / rays under coord_polar
                pieces, _, _ = munched_segment(coord, panel_params, x0[i], y0[i], x1[i], y1[i])
                if not pieces:
                    continue
                line = multi_path_mobject(pieces, color, width, alpha)
                q = pieces[-1][-1]
            else:
                line = paint_stroke(Line(a[i], b[i]), color, width, alpha)
                q = b[i]
            lines.add(apply_linetype(line, row.get("linetype")))
            drawn.append(i)
            text = group_label(label, i, i)
            if self.vertical:  # reading upwards, along the top-left of the line
                lab = direct_label(
                    theme, text, color, q, hjust=1, vjust=0, angle=90, nudge=(-0.05, 0)
                )
            else:  # above the right end
                lab = direct_label(theme, text, color, q, hjust=1, vjust=0, nudge=(0, 0.05))
            if lab is not None:
                labels.add(lab)
        # one line per row, so marks are per row (each intercept is its own mark)
        return attach_marks(assemble(lines, labels), data.iloc[drawn], per="row")

    def _radar_spokes(self, coord, panel_params) -> np.ndarray | None:
        """Theta breaks (the spokes) when this line runs along theta of a
        ``coord_radar`` (hline with ``theta="x"``, vline with ``theta="y"``)."""
        theta = getattr(coord, "theta", None)
        if theta is None or not getattr(coord, "is_linear", False) or panel_params is None:
            return None
        if (theta == "x") == self.vertical:
            return None
        br = np.asarray(getattr(panel_params, f"{theta}_breaks", ()), dtype=float)
        br = np.unique(br[np.isfinite(br)])
        return br if len(br) >= 2 else None

    def _ring(self, spokes: np.ndarray, v: float) -> tuple[np.ndarray, np.ndarray]:
        """Closed ``(x, y)`` ring through ``spokes`` at intercept ``v``."""
        t = np.append(spokes, spokes[0])
        c = np.full_like(t, v)
        return (c, t) if self.vertical else (t, c)

    def draw_key(self, data_row, theme: Theme | None = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        row = self.key_aes(data_row)
        a, b = (DOWN, UP) if self.vertical else (LEFT, RIGHT)
        # hline keys widen in linetype legends; vline keys keep the cell height
        w = KEY_SIZE if self.vertical else getattr(theme, "legend_key_width", None) or KEY_SIZE
        line = paint_stroke(
            Line(a * w / 2, b * w / 2),
            resolve_color(row.get("color"), theme.get_foreground()),
            num(row.get("linewidth"), 2.0),
            num(row.get("alpha"), 1.0),
        )
        return apply_linetype(line, row.get("linetype"), max_period=w / 2)


class GeomVline(_RefLine):
    """Vertical reference lines; only ``xintercept`` trains the x scale."""

    required_aes = ("xintercept",)
    x_aes = ("xintercept",)
    y_aes = ()
    vertical = True


class GeomHline(_RefLine):
    """Horizontal reference lines; only ``yintercept`` trains the y scale."""

    required_aes = ("yintercept",)
    x_aes = ()
    y_aes = ("yintercept",)
    vertical = False


def _ref_layer(geom, col, value, mapping, data, position, name, params):
    if value is not None:
        if mapping is not None or data is not None:
            raise TypeError(f"pass either {col}= or mapping/data, not both")
        try:
            values = np.atleast_1d(np.asarray(value, dtype=float))
        except (TypeError, ValueError):  # dates / ISO strings for a date scale
            values = np.atleast_1d(np.asarray(value, dtype=object))
        data = pd.DataFrame({col: values})
        mapping = aes(**{col: col})
    # ggplot2: reference lines never inherit the plot's aesthetics.
    return make_layer(
        geom,
        mapping,
        data,
        stat="identity",
        position=position,
        inherit_aes=False,
        name=name,
        params=params,
    )


@register
def geom_vline(
    mapping=None,
    data=None,
    *,
    xintercept=None,
    position="identity",
    name: str | None = None,
    **params,
):
    """Vertical line(s) at ``xintercept`` (a number or list, in data space).

    With ``xintercept=`` the layer carries its own data
    (``DataFrame({"xintercept": ...})`` mapped by ``aes(xintercept="xintercept")``)
    and never inherits the plot's aesthetics; alternatively map
    ``aes(xintercept=column)``. ``label=`` and constant ``color``/``linewidth``/
    ``alpha`` go in ``params``.
    """
    return _ref_layer(GeomVline(), "xintercept", xintercept, mapping, data, position, name, params)


@register
def geom_hline(
    mapping=None,
    data=None,
    *,
    yintercept=None,
    position="identity",
    name: str | None = None,
    **params,
):
    """Horizontal line(s) at ``yintercept``; see [`geom_vline()`][ggmanim.geoms.geom_vline]."""
    return _ref_layer(GeomHline(), "yintercept", yintercept, mapping, data, position, name, params)
