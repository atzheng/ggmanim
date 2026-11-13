"""Guides: legends for discrete non-position scales.

:func:`build_legends` turns the trained non-position scales of a plot into
:class:`Legend` mobjects (title + one row per break: key glyph + label). Key
glyphs come from the geoms that map the legend's aesthetic(s)
(:meth:`ggmanim.geoms.base.Geom.draw_key`); if several layers map it, their
glyphs are overlaid in one key, as in ggplot2. Legends of different aesthetics
with the same title, breaks and labels (e.g. ``color`` and ``fill`` both mapped
to one column) are merged into one legend whose keys carry all of them.

:func:`place_legends` positions the legends next to the panel according to
``theme.legend_position``. :func:`guides` is the component that switches
legends off per aesthetic: ``p.add(guides(color="none"))``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Mobject,
    Square,
    SurroundingRectangle,
    VGroup,
    VMobject,
)

from .aes import AfterStat, standardise_aes_name
from .geoms.base import KEY_SIZE
from .registry import register
from .theme import Theme, theme_default

__all__ = ["Legend", "Guides", "build_legends", "place_legends", "guides"]

#: Valid guide specs for :func:`guides`.
GUIDE_TYPES = ("legend", "none")

#: Horizontal gap between a key and its label, and between title and rows.
_KEY_LABEL_BUFF = 0.15
_TITLE_BUFF = 0.15
_ROW_GAP = 0.08
#: Gap between legend and panel, and between stacked legends.
_PANEL_BUFF = 0.4
_LEGEND_SPACING = 0.4

_DIRECTIONS = {"right": RIGHT, "left": LEFT, "top": UP, "bottom": DOWN}


class Legend(VGroup):
    """A legend: ``VGroup(title?, *rows)``.

    Attributes
    ----------
    title : Mobject or None
    keys : list of VGroup
        One key per break; each is a ``VGroup`` of the overlaid glyphs (one per
        contributing layer), centred on the key cell.
    labels : list of Mobject
        One label per break (``theme.make_text``).
    rows : list of VGroup
        ``VGroup(key, label)`` per break, in break order.
    aesthetics : tuple of str
        The (merged) aesthetics this legend explains, e.g. ``("color", "fill")``.
    breaks : list
        The scale breaks (levels) shown.
    """

    def __init__(
        self,
        title: Mobject | None,
        keys: list[VGroup],
        labels: list[Mobject],
        aesthetics: tuple[str, ...],
        breaks: Sequence[Any] = (),
        theme: Theme | None = None,
    ):
        self.title = title
        self.keys = list(keys)
        self.labels = list(labels)
        self.rows = [VGroup(k, l) for k, l in zip(self.keys, self.labels)]  # type: ignore[arg-type]
        self.aesthetics = tuple(aesthetics)
        self.breaks = list(breaks)
        self.box: VMobject | None = None
        parts = ([title] if title is not None else []) + self.rows
        super().__init__(*parts)  # type: ignore[arg-type]
        self.arrange_rows("vertical")
        theme = theme if theme is not None else theme_default()
        if getattr(theme, "legend_box", False):
            self.box = SurroundingRectangle(
                VGroup(*parts),  # type: ignore[arg-type]
                color=theme.get_foreground(),
                buff=0.15,
                stroke_width=1,
            )
            self.add(self.box)

    def arrange_rows(self, direction: str = "vertical") -> Legend:
        """Lay out the rows vertically (default; right/left legends) or
        horizontally (top/bottom legends); the title goes above, left-aligned."""
        cell = KEY_SIZE * 1.2
        x = 0.0
        y = 0.0
        for row in self.rows:
            key, label = row
            row_h = max(cell, label.height)
            if direction == "horizontal":
                key.move_to((x + cell / 2, 0.0, 0.0))
                label.move_to(ORIGIN)
                label.align_to((x + cell + _KEY_LABEL_BUFF, 0.0, 0.0), LEFT)
                x += cell + _KEY_LABEL_BUFF + label.width + 2 * _KEY_LABEL_BUFF
            else:
                key.move_to((cell / 2, y - row_h / 2, 0.0))
                label.move_to((0.0, y - row_h / 2, 0.0))
                label.align_to((cell + _KEY_LABEL_BUFF, 0.0, 0.0), LEFT)
                y -= row_h + _ROW_GAP
        if self.title is not None and self.rows:
            rows = VGroup(*self.rows)
            self.title.next_to(rows, UP, buff=_TITLE_BUFF, aligned_edge=LEFT)
        if self.box is not None:
            content = VGroup(*([self.title] if self.title is not None else []), *self.rows)  # type: ignore[arg-type]
            self.box.become(
                SurroundingRectangle(content, color=self.box.get_color(), buff=0.15, stroke_width=1)
            )
        return self


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------


def _as_layer_tuple(layer: Any) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    """``(geom, resolved_mapping, params)``; a bare ``Layer`` uses its own mapping."""
    if isinstance(layer, tuple):
        geom, mapping, params = layer
    else:
        geom, mapping, params = layer.geom, layer.mapping, layer.params
    mapping = {standardise_aes_name(k): v for k, v in dict(mapping or {}).items()}
    return geom, mapping, dict(params or {})


def _column_name(value: Any) -> str | None:
    """A printable name for a mapping value (column name / after_stat expr)."""
    if isinstance(value, AfterStat):
        value = value.expr
    if isinstance(value, str):
        return value
    return None


def _scale_title(scale, aes: str, labels: Mapping[str, Any], mapping_cols: Mapping[str, str]):
    """Title priority: ``scale.name``, ``labels[aes]`` (``None`` there = no title),
    the mapped column name, else ``None``."""
    if getattr(scale, "name", None) is not None:
        return scale.name
    if aes in labels:
        return labels[aes]
    return mapping_cols.get(aes)


def _title_key(title: Any) -> Any:
    if title is None or isinstance(title, str):
        return ("s", title)
    return ("id", id(title))


def _guide_off(aes: str, scale, guides_spec: Mapping[str, Any]) -> bool:
    spec = guides_spec.get(aes)
    if spec is None:
        spec = getattr(scale, "guide", "legend")
    return spec in ("none", False)


def _show_legend(params: Mapping[str, Any]) -> bool | None:
    v = params.get("show_legend", params.get("show.legend"))
    return None if v is None else bool(v)


def build_legends(
    scales: Mapping[str, Any],
    layers: Iterable[Any],
    theme: Theme | None = None,
    labels: Mapping[str, Any] | None = None,
    guides: Mapping[str, Any] | None = None,
) -> list[Legend]:
    """Build one :class:`Legend` per (merged) discrete non-position scale.

    Parameters
    ----------
    scales:
        ``aesthetic -> trained scale``. Scales that are not discrete
        (``is_discrete`` falsy), whose ``guide`` is ``"none"``, or that have no
        breaks are skipped. The same scale object may appear under several keys.
    layers:
        ``(geom, resolved_mapping, params)`` per layer (``Layer`` objects are
        also accepted; their own, unresolved mapping is used). A layer
        contributes a glyph to a legend when it maps one of the legend's
        aesthetics, unless ``params["show_legend"]`` is false;
        ``show_legend=True`` forces a glyph.
    labels:
        ``labs()`` labels (aesthetic -> str/Mobject/None).
    guides:
        ``guides()`` spec (aesthetic -> ``"legend"``/``"none"``); overrides
        ``scale.guide``.

    Returns the legends in scale order, each laid out vertically at the origin.
    """
    theme = theme if theme is not None else theme_default()
    labels = {standardise_aes_name(k): v for k, v in dict(labels or {}).items()}
    guides_spec = {standardise_aes_name(k): v for k, v in dict(guides or {}).items()}
    layer_tuples = [_as_layer_tuple(l) for l in layers]

    mapping_cols: dict[str, str] = {}
    for _, mapping, _ in layer_tuples:
        for aes, v in mapping.items():
            name = _column_name(v)
            if name is not None:
                mapping_cols.setdefault(aes, name)

    # group aesthetics into legends by (title, breaks, labels)
    groups: dict[Any, dict[str, Any]] = {}
    for aes, scale in scales.items():
        aes = standardise_aes_name(aes)
        if aes in ("x", "y") or not getattr(scale, "is_discrete", False):
            continue
        if _guide_off(aes, scale, guides_spec):
            continue
        breaks = list(scale.get_breaks())
        if not breaks:
            continue
        blabels = [str(b) for b in scale.get_labels()]
        title = _scale_title(scale, aes, labels, mapping_cols)
        key = (_title_key(title), tuple(map(repr, breaks)), tuple(blabels))
        g = groups.setdefault(key, {"title": title, "breaks": breaks, "labels": blabels, "aes": {}})
        g["aes"].setdefault(aes, scale)

    legends = []
    for g in groups.values():
        legends.append(_make_legend(g, layer_tuples, theme))
    return legends


def _make_legend(group: Mapping[str, Any], layer_tuples, theme: Theme) -> Legend:
    aes_scales: dict[str, Any] = group["aes"]
    aesthetics = tuple(aes_scales)
    breaks = group["breaks"]
    mapped = {aes: list(scale.map(breaks)) for aes, scale in aes_scales.items()}

    contributing = []
    for geom, mapping, params in layer_tuples:
        show = _show_legend(params)
        if show is False:
            continue
        used = tuple(a for a in aesthetics if a in mapping)
        if used or show:
            contributing.append((geom, used if used else aesthetics, params))

    fg = theme.get_foreground()
    keys = []
    for i, _ in enumerate(breaks):
        glyphs = VGroup()
        for geom, used, params in contributing:
            row = {a: mapped[a][i] for a in used}
            try:
                row.update(geom.aes_params(params))
            except AttributeError:
                pass
            try:
                glyph = geom.draw_key(row, theme)
            except NotImplementedError:
                continue
            if glyph is not None:
                glyphs.add(glyph.move_to(ORIGIN))
        if len(glyphs) == 0:  # no geom draws a key: plain filled square
            colour = next((mapped[a][i] for a in ("fill", "color") if a in mapped), fg)
            glyphs.add(
                Square(side_length=KEY_SIZE, fill_color=colour, fill_opacity=1, stroke_width=0)
            )
        keys.append(glyphs)

    labels = [theme.make_text(lab, font_size=theme.legend_text_size) for lab in group["labels"]]
    title = group["title"]
    if title is not None:
        if isinstance(title, Mobject):
            title = title.copy()
        else:
            title = theme.make_text(str(title), font_size=theme.legend_title_size)
    return Legend(title, keys, labels, aesthetics, breaks=breaks, theme=theme)


# ---------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------


def place_legends(
    panel_mobject: Mobject, legends: Sequence[Legend], theme: Theme | None = None
) -> VGroup:
    """Stack ``legends`` and put them beside ``panel_mobject``.

    ``theme.legend_position`` ``"right"``/``"left"``: legends stacked vertically,
    vertically centred on the panel. ``"top"``/``"bottom"``: rows laid out
    horizontally, legends side by side, horizontally centred. ``"none"`` (or no
    legends): an empty ``VGroup``. Only the legends are moved; the panel is not
    touched.
    """
    theme = theme if theme is not None else theme_default()
    position = theme.legend_position
    if position == "none" or not legends:
        return VGroup()
    horizontal = position in ("top", "bottom")
    for leg in legends:
        leg.arrange_rows("horizontal" if horizontal else "vertical")
    group = VGroup(*legends)
    if horizontal:
        group.arrange(RIGHT, buff=_LEGEND_SPACING, aligned_edge=UP)
    else:
        group.arrange(DOWN, buff=_LEGEND_SPACING, aligned_edge=LEFT)
    direction = _DIRECTIONS[position]
    group.next_to(panel_mobject, direction, buff=_PANEL_BUFF)
    return group


# ---------------------------------------------------------------------------
# component
# ---------------------------------------------------------------------------


class Guides:
    """Component holding ``aesthetic -> guide`` (``"legend"`` / ``"none"``)."""

    __slots__ = ("guides",)

    def __init__(self, **kw):
        out = {}
        for k, v in kw.items():
            if v is None or v is False:
                v = "none"
            if v not in GUIDE_TYPES:
                raise ValueError(f"guide for {k!r} must be one of {GUIDE_TYPES}, got {v!r}")
            out[standardise_aes_name(k)] = v
        self.guides: dict[str, str] = out

    def add_to(self, plot):
        return plot._with_guides(self.guides)

    def __repr__(self):
        return f"guides({', '.join(f'{k}={v!r}' for k, v in self.guides.items())})"


@register
def guides(**kw) -> Guides:
    """Set guides per aesthetic: ``guides(color="none")`` hides the colour legend;
    ``"legend"`` restores it (``None``/``False`` mean ``"none"``)."""
    return Guides(**kw)
