"""geom_path / geom_line: connect observations (in data order / in order of x),
one path per group."""

from __future__ import annotations

from manim import LEFT, RIGHT, Line, VGroup, VMobject

from ..registry import register
from ..theme import Theme, theme_default
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
    make_label,
    make_layer,
    multi_path_mobject,
    num,
    resolve_color,
)

__all__ = ["GeomPath", "GeomLine", "geom_path", "geom_line"]


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
    """

    required_aes = ("x", "y")
    default_aes = {"color": None, "linewidth": 3.0, "alpha": 1.0, "linetype": "solid"}
    extra_params: tuple[str, ...] = ("label",)
    sort_by_x = False

    def draw_panel(
        self, data, panel_params, coord, theme: Theme | None = None, label=None, **params
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        self.check_required_aes(data)
        data = finite_rows(self.use_defaults(data, params), ("x", "y"))
        fg = theme.get_foreground()
        paths, labels = VGroup(), VGroup()
        for i, (gid, gdf) in enumerate(data.groupby("group", sort=True)):
            if self.sort_by_x:
                gdf = gdf.sort_values("x", kind="stable")
            pieces = clipped_path_points(
                coord, panel_params, gdf["x"].to_numpy(float), gdf["y"].to_numpy(float)
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
            lab = make_label(theme, group_label(label, i, gid), color)
            if lab is not None and pieces:
                labels.add(lab.next_to(pieces[-1][-1], RIGHT, buff=0.1))
        return attach_marks(assemble(paths, labels), data, per="group")

    def draw_key(self, data_row, theme: Theme | None = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        row = self.key_aes(data_row)
        w = getattr(theme, "legend_key_width", None) or KEY_SIZE  # wider for linetype legends
        line = Line(
            LEFT * w / 2,
            RIGHT * w / 2,
            color=resolve_color(row.get("color"), theme.get_foreground()),
            stroke_width=num(row.get("linewidth"), 3.0),
            stroke_opacity=num(row.get("alpha"), 1.0),
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
    group id -> label) is drawn at each group's last point in the group's colour.
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
    group id -> label) is drawn at each group's last point in the group's colour.
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
