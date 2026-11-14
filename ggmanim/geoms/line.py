"""geom_path / geom_line: connect observations (in data order / in order of x),
one path per group."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from manim import LEFT, RIGHT, Line, VGroup, VMobject

from ..registry import register
from ..theme import Theme, theme_default
from ._labels import direct_label
from ._linetype import apply_linetype
from ._segments import segmented_path, varies_within
from .base import (
    KEY_SIZE,
    Geom,
    assemble,
    attach_marks,
    clipped_path_points,
    finite_rows,
    first,
    group_label,
    make_layer,
    multi_path_mobject,
    munch_path,
    num,
    paint_stroke,
    resolve_color,
)

__all__ = ["GeomPath", "GeomLine", "geom_path", "geom_line"]


def squish_infinite(data: pd.DataFrame, panel_params) -> pd.DataFrame:
    """``x``/``y`` values of ``+-inf`` moved onto the panel edges
    (``panel_params.x_range``/``y_range``), like ggplot2's coord squishing
    infinite values. Other values (NaN included) are unchanged."""
    out = {}
    for col in ("x", "y"):
        rng = getattr(panel_params, f"{col}_range", None)
        if col not in data.columns or rng is None or not len(data):
            continue
        v = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
        if not np.isinf(v).any():
            continue
        lo, hi = float(min(rng[0], rng[1])), float(max(rng[0], rng[1]))
        out[col] = np.where(v == -np.inf, lo, np.where(v == np.inf, hi, v))
    return data.assign(**out) if out else data


class GeomPath(Geom):
    """Connect observations in data order, one ``VMobject`` per group.

    ``draw_panel`` returns ``VGroup(*paths, *labels)``: one open polyline per
    group (in ``group`` order; vertex count == number of finite rows in the
    group; a ``VGroup`` of dashes for a non-solid ``linetype``, see
    :func:`~ggmanim.geoms._linetype.apply_linetype`; when colour, linewidth or
    alpha vary within the group, a ``VGroup`` of per-segment strokes, see
    :func:`~ggmanim.geoms._segments.segmented_path`), then — when ``label=`` is
    given — one label per group placed right of the group's last visible point
    in the group's colour (for a per-segment group, the colour of its last
    visible segment). ``.marks`` holds the paths (one per group), ``.keys`` the
    group ids (or ``key`` values) and ``.labels`` the labels.

    ``x``/``y`` of ``+-inf`` are kept and drawn on the panel edges
    (:func:`squish_infinite`); only NaN positions are dropped.
    """

    required_aes = ("x", "y")
    default_aes = {"color": None, "linewidth": 3.0, "alpha": 1.0, "linetype": "solid"}
    extra_params: tuple[str, ...] = ("label",)
    sort_by_x = False

    def handle_na(
        self, data: pd.DataFrame, params: Mapping[str, Any] | None = None, label: str | None = None
    ) -> pd.DataFrame:
        # +-inf are kept (drawn on the panel edges); only NaN positions are missing
        if not len(data):
            return data
        finite = {
            c: pd.to_numeric(data[c], errors="coerce").replace([np.inf, -np.inf], 0.0)
            for c in ("x", "y")
            if c in data.columns
        }
        kept = super().handle_na(data.assign(**finite), params, label)
        return data.loc[kept.index]

    def draw_panel(
        self, data, panel_params, coord, theme: Theme | None = None, label=None, **params
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        self.check_required_aes(data)
        data = squish_infinite(data, panel_params)
        data = finite_rows(self.use_defaults(data, params), ("x", "y"))
        fg = theme.get_foreground()
        paths, labels = VGroup(), VGroup()
        for i, (gid, gdf) in enumerate(data.groupby("group", sort=True)):
            if self.sort_by_x:
                gdf = gdf.sort_values("x", kind="stable")
            pieces = clipped_path_points(
                coord, panel_params, *munch_path(coord, gdf["x"], gdf["y"])
            )
            color = resolve_color(first(gdf, "color"), fg)
            if varies_within(gdf):  # ggplot: one segment per vertex pair, styled by its start
                segs = segmented_path(coord, panel_params, gdf, fg)
                if len(segs):  # label in the colour of the last visible segment
                    color = segs[-1].get_stroke_color()
                paths.add(segs)
            else:
                # clip first, then dash each visible subpath
                path = multi_path_mobject(
                    pieces, color, num(first(gdf, "linewidth"), 3.0), num(first(gdf, "alpha"), 1.0)
                )
                paths.add(apply_linetype(path, first(gdf, "linetype")))
            if pieces:  # direct label right of the last visible point
                lab = direct_label(
                    theme, group_label(label, i, gid), color, pieces[-1][-1], nudge=(0.1, 0.0)
                )
                if lab is not None:
                    labels.add(lab)
        return attach_marks(assemble(paths, labels), data, per="group")

    def draw_key(self, data_row, theme: Theme | None = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        row = self.key_aes(data_row)
        w = getattr(theme, "legend_key_width", None) or KEY_SIZE  # wider for linetype legends
        line = paint_stroke(
            Line(LEFT * w / 2, RIGHT * w / 2),
            resolve_color(row.get("color"), theme.get_foreground()),
            num(row.get("linewidth"), 3.0),
            num(row.get("alpha"), 1.0),
        )
        return apply_linetype(line, row.get("linetype"), max_period=w / 2)


class GeomLine(GeomPath):
    """Like :class:`GeomPath` but each group is sorted by x first."""

    sort_by_x = True


@register
def geom_path(
    mapping=None,
    data=None,
    *,
    stat="identity",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Paths connecting observations in the order they appear in the data, one
    per group (unlike :func:`geom_line`, which sorts each group by x first).

    ``label=`` (str or Mobject for every group, a list in group order, or a dict
    group value or id -> label) is drawn at each group's last point in the group's colour
    (a *direct label*, drawn like ``geom_text``; see :mod:`ggmanim.geoms._labels`).
    """
    return make_layer(
        GeomPath(),
        mapping,
        data,
        stat=stat,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )


@register
def geom_line(
    mapping=None,
    data=None,
    *,
    stat="identity",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Lines connecting observations in order of x, one per group.

    ``label=`` (str or Mobject for every group, a list in group order, or a dict
    group value or id -> label) is drawn at each group's last point in the group's colour
    (a *direct label*, drawn like ``geom_text``; see :mod:`ggmanim.geoms._labels`).
    """
    return make_layer(
        GeomLine(),
        mapping,
        data,
        stat=stat,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
