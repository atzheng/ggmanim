"""Guides: legends for discrete non-position scales (and continuous ones that ask
for a binned legend, e.g. size / alpha / linewidth); colourbars for continuous
colour scales live in :mod:`.guides_colourbar`.

:func:`build_legends` turns the trained non-position scales of a plot into
:class:`Legend` mobjects (title + one row per break: key glyph + label). Key
glyphs come from the geoms that map the legend's aesthetic(s)
(:meth:`ggmanim.geoms.base.Geom.draw_key`); if several layers map it, their
glyphs are overlaid in one key, as in ggplot2. Legends of different aesthetics
with the same title, breaks and labels (e.g. ``color`` and ``fill`` both mapped
to one column) are merged into one legend whose keys carry all of them.
Options per legend come from :func:`~ggmanim.guides_legend.guide_legend`
(title, ``nrow``/``ncol``/``byrow`` key grid, ``reverse``, ``override_aes``,
``order``, ``direction``); guides are returned sorted by ``order``.

:func:`place_legends` lays the legends out (``theme.legend_direction``,
``legend_box``, ``legend_spacing``) and positions them next to the panel, or
inside it for an ``(x, y)`` ``theme.legend_position``, honouring
``legend_justification``. :func:`guides` is the component that sets guides per
aesthetic: ``p.add(guides(color="none", shape=guide_legend(ncol=2)))``.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from manim import (
    DL,
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    UR,
    Mobject,
    Square,
    VGroup,
)

from .aes import AfterStat, standardise_aes_name
from .geoms.base import KEY_SIZE
from .guides_colourbar import (
    Colourbar,
    GuideColourbar,
    build_colourbar,
    guide_colorbar,
    guide_colourbar,
    is_colourbar_spec,
)
from .guides_coloursteps import (
    Coloursteps,
    GuideColoursteps,
    build_coloursteps,
    guide_colorsteps,
    guide_coloursteps,
    is_coloursteps_spec,
)
from .guides_legend import GuideLegend, guide_legend, is_legend_spec, legend_frame
from .labels import render_label
from .registry import register
from .theme import Theme, theme_default

__all__ = [
    "Legend",
    "Guides",
    "build_legends",
    "place_legends",
    "guides",
    "Colourbar",
    "GuideColourbar",
    "guide_colorbar",
    "guide_colourbar",
    "Coloursteps",
    "GuideColoursteps",
    "guide_colorsteps",
    "guide_coloursteps",
    "GuideLegend",
    "guide_legend",
]

#: Valid guide specs for :func:`guides`.
GUIDE_TYPES = ("legend", "none", "colourbar", "colorbar", "coloursteps", "colorsteps")
#: Guide objects accepted alongside :data:`GUIDE_TYPES`.
GUIDE_CLASSES = (GuideLegend, GuideColourbar, GuideColoursteps)

#: Horizontal gap between a key and its label, and between title and rows.
_KEY_LABEL_BUFF = 0.15
_TITLE_BUFF = 0.15
_ROW_GAP = 0.08
_COL_GAP = 0.3
#: Line-key length in legends showing ``linetype`` (unless ``legend_key_width`` is set).
_LINETYPE_KEY_WIDTH = KEY_SIZE * 2.5
#: Gap between legend and panel, and between stacked legends.
_PANEL_BUFF = 0.4
_LEGEND_SPACING = 0.4

_DIRECTIONS = {"right": RIGHT, "left": LEFT, "top": UP, "bottom": DOWN}


class Legend(VGroup):
    """A legend: ``VGroup(box?, title?, *rows)``.

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
    options : GuideLegend
        The :func:`~ggmanim.guides_legend.guide_legend` options.
    direction : ``"vertical"`` or ``"horizontal"`` (current layout).
    nrow, ncol : int
        The key grid of the current layout.
    box : Mobject or None
        The background / border (``theme(legend_background=...,
        legend_border=True)``), behind everything else.
    """

    def __init__(
        self,
        title: Mobject | None,
        keys: list[VGroup],
        labels: list[Mobject],
        aesthetics: tuple[str, ...],
        breaks: Sequence[Any] = (),
        theme: Theme | None = None,
        options: GuideLegend | None = None,
        direction: str = "vertical",
    ):
        self.title = title
        self.keys = list(keys)
        self.labels = list(labels)
        self.rows = [VGroup(k, l) for k, l in zip(self.keys, self.labels)]  # type: ignore[arg-type]
        self.aesthetics = tuple(aesthetics)
        self.breaks = list(breaks)
        self.options = options if options is not None else GuideLegend()
        self.theme = theme if theme is not None else theme_default()
        self.box: Mobject | None = None
        self.direction = direction
        self.nrow, self.ncol = 1, 1
        super().__init__()
        self.arrange_rows(direction)

    def arrange_rows(self, direction: str = "vertical") -> Legend:
        """Lay out the keys in a grid (``guide_legend(nrow=, ncol=, byrow=)``;
        default one column for ``"vertical"``, one row for ``"horizontal"``), the
        title above aligned by ``theme.legend_title_align``, and the optional
        background / border behind."""
        self.direction = "horizontal" if direction == "horizontal" else "vertical"
        theme = self.theme
        n = len(self.rows)
        nrow, ncol = self.options.grid(n, self.direction)
        self.nrow, self.ncol = nrow, ncol
        cells = [divmod(i, ncol) if self.options.byrow else divmod(i, nrow)[::-1] for i in range(n)]
        key_size = float(getattr(theme, "legend_key_size", KEY_SIZE * 1.2))
        # uniform key cells; they grow for big keys (e.g. scale_size_area(max_size=...))
        key_w = max([key_size] + [k.width for k in self.keys])
        key_h = max([key_size] + [k.height for k in self.keys])
        col_w = [key_w] * ncol
        row_h = [key_h] * nrow
        for (r, c), label in zip(cells, self.labels):
            col_w[c] = max(col_w[c], key_w + _KEY_LABEL_BUFF + label.width)
            row_h[r] = max(row_h[r], label.height)
        col_x = [sum(col_w[:c]) + c * _COL_GAP for c in range(ncol)]
        row_y = [-(sum(row_h[:r]) + r * _ROW_GAP) for r in range(nrow)]
        for (r, c), key, label in zip(cells, self.keys, self.labels):
            x, y = col_x[c], row_y[r] - row_h[r] / 2
            key.move_to((x + key_w / 2, y, 0.0))
            label.move_to((0.0, y, 0.0))
            label.align_to((x + key_w + _KEY_LABEL_BUFF, 0.0, 0.0), LEFT)
        parts: list[Any] = list(self.rows)
        if self.title is not None:
            if self.rows:
                rows = VGroup(*self.rows)
                self.title.next_to(rows, UP, buff=_TITLE_BUFF)
                align = float(getattr(theme, "legend_title_align", 0.0))
                left = rows.get_left()[0] + align * (rows.width - self.title.width)
                self.title.set_x(left, direction=LEFT)
            parts.insert(0, self.title)
        self.box = legend_frame(VGroup(*parts), theme) if parts else None
        if self.box is not None:
            parts.insert(0, self.box)
        self.submobjects = []
        self.add(*parts)
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


def _binned_legend(aes: str, scale, guides_spec: Mapping[str, Any]) -> bool:
    """A continuous scale whose guide is ``"legend"`` or a :func:`guide_legend`
    (the default of the size / alpha / linewidth range scales, or
    ``guides(aes="legend")``): keys at its breaks."""
    spec = guides_spec.get(aes, getattr(scale, "guide", None))
    return is_legend_spec(spec) and hasattr(scale, "trans")


def _legend_breaks(scale) -> list:
    """Legend breaks: discrete levels, or a continuous scale's breaks in *data*
    space (so ``scale.map`` gives the key values)."""
    breaks = scale.get_breaks()
    if getattr(scale, "is_discrete", False):
        return list(breaks)
    return [float(b) for b in scale.trans.inverse(breaks)]


def _guide_spec(aes: str, scale, guides_spec: Mapping[str, Any]) -> Any:
    spec = guides_spec.get(aes)
    return spec if spec is not None else getattr(scale, "guide", "legend")


def _show_legend(params: Mapping[str, Any]) -> Any:
    """``show_legend`` of a layer: ``None``, a bool, or a dict aesthetic -> bool."""
    v = params.get("show_legend", params.get("show.legend"))
    if v is None:
        return None
    if isinstance(v, Mapping):
        return {standardise_aes_name(k): (None if b is None else bool(b)) for k, b in v.items()}
    return bool(v)


def _include_layer(show: Any, matched: tuple[str, ...], aesthetics: tuple[str, ...]) -> bool:
    """ggplot2's rule: a layer mapping some of the legend's aesthetics is drawn
    unless every explicit ``show_legend`` value for them is false; a layer
    mapping none of them only when forced (``True`` or a true dict entry)."""
    if isinstance(show, dict):
        if matched:
            explicit = [show[a] for a in matched if show.get(a) is not None]
            return not explicit or any(explicit)
        return any(show.get(a) is True for a in aesthetics)
    if show is None:
        return bool(matched)
    return bool(show)


def _guide_order(guide: Any) -> int:
    order = getattr(getattr(guide, "options", None), "order", 0)
    return order if order > 0 else 99


def build_legends(
    scales: Mapping[str, Any],
    layers: Iterable[Any],
    theme: Theme | None = None,
    labels: Mapping[str, Any] | None = None,
    guides: Mapping[str, Any] | None = None,
) -> list[Legend]:
    """Build one guide per (merged) non-position scale: a :class:`Legend` for
    discrete scales (and continuous ones whose guide is ``"legend"`` or a
    :func:`guide_legend`: one key per break), a
    :class:`~ggmanim.guides_colourbar.Colourbar` for continuous scales whose
    guide is ``"colourbar"`` or a :func:`guide_colorbar`.

    Parameters
    ----------
    scales:
        ``aesthetic -> trained scale``. Continuous scales with a legend guide
        (the size/alpha/linewidth range scales) get keys at ``get_breaks()``;
        continuous colour scales with a colourbar guide get a colourbar; other
        non-discrete scales, scales whose ``guide`` is ``"none"``, and scales
        without breaks are skipped. The same scale object may appear under
        several keys.
    layers:
        ``(geom, resolved_mapping, params)`` per layer (``Layer`` objects are
        also accepted; their own, unresolved mapping is used). A layer
        contributes a glyph to a legend when it maps one of the legend's
        aesthetics, unless ``params["show_legend"]`` is false (or, as a dict
        ``{aes: bool}``, false for every one of them it maps);
        ``show_legend=True`` forces a glyph. A legend whose mapping layers are
        all excluded is dropped.
    labels:
        ``labs()`` labels (aesthetic -> str/Mobject/None).
    guides:
        ``guides()`` spec (aesthetic -> ``"legend"``/``"colourbar"``/``"none"``,
        a :func:`guide_legend` or a :func:`guide_colorbar`); overrides
        ``scale.guide``.

    Legends whose kind, title, breaks and labels match are merged (ggplot2's
    rule), e.g. ``color`` and ``shape`` mapped to one column give one legend with
    composite keys. Returns the guides sorted by their ``order`` option (``0`` =
    after the ordered ones, in scale order), each laid out vertically at the origin.
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

    # group aesthetics into guides by (kind, title, breaks, labels)
    groups: dict[Any, dict[str, Any]] = {}
    for aes, scale in scales.items():
        aes = standardise_aes_name(aes)
        if aes in ("x", "y"):
            continue
        if _guide_off(aes, scale, guides_spec):
            continue
        discrete = getattr(scale, "is_discrete", False)
        spec = _guide_spec(aes, scale, guides_spec)
        steps = not discrete and is_coloursteps_spec(spec)
        colourbar = steps or (not discrete and is_colourbar_spec(spec))
        if not discrete and not colourbar and not _binned_legend(aes, scale, guides_spec):
            continue
        breaks = list(scale.get_breaks()) if colourbar else _legend_breaks(scale)
        if not breaks:
            continue
        blabels = [str(b) for b in scale.get_labels()]
        opts = spec if isinstance(spec, GUIDE_CLASSES) else None
        if isinstance(opts, GuideLegend) and opts.reverse:
            breaks, blabels = breaks[::-1], blabels[::-1]
        title = _scale_title(scale, aes, labels, mapping_cols)
        if opts is not None and opts.title is not None:
            title = opts.title
        kind: tuple[Any, ...]
        if colourbar:
            kind = ("coloursteps" if steps else "colourbar", repr(spec))
        else:
            kind = ("legend", getattr(opts, "direction", None))
        key = (kind, _title_key(title), tuple(map(repr, breaks)), tuple(blabels))
        g = groups.setdefault(key, {"title": title, "breaks": breaks, "labels": blabels, "aes": {}})
        g["aes"].setdefault(aes, scale)
        if colourbar:
            if steps:
                g["colourbar"] = spec if isinstance(spec, GuideColoursteps) else GuideColoursteps()
            else:
                g["colourbar"] = spec if isinstance(spec, GuideColourbar) else GuideColourbar()
        else:
            legend_opts = opts if isinstance(opts, GuideLegend) else None
            g["options"] = _merge_options(g.get("options"), legend_opts)

    legends: list[Any] = []
    for g in groups.values():
        if isinstance(g.get("colourbar"), GuideColoursteps):
            legends.append(
                build_coloursteps(
                    g["title"], g["aes"], g["breaks"], g["labels"], theme, g["colourbar"]
                )
            )
        elif "colourbar" in g:
            legends.append(
                build_colourbar(
                    g["title"], g["aes"], g["breaks"], g["labels"], theme, g["colourbar"]
                )
            )
        else:
            leg = _make_legend(g, layer_tuples, theme)
            if leg is not None:
                legends.append(leg)
    legends.sort(key=_guide_order)
    return legends


def _merge_options(old: GuideLegend | None, new: GuideLegend | None) -> GuideLegend:
    """Options of a merged legend: the first legend's, with ``override_aes``
    combined (first wins per aesthetic) and the first non-zero ``order``."""
    if old is None:
        return copy.copy(new) if new is not None else GuideLegend()
    if new is not None:
        old.override_aes = {**new.override_aes, **old.override_aes}
        if old.order == 0:
            old.order = new.order
    return old


def _override(value: Any, i: int) -> Any:
    """An ``override_aes`` value for key ``i`` (a list gives one value per key)."""
    if isinstance(value, list):
        return value[i % len(value)] if value else None
    return value


def _make_legend(group: Mapping[str, Any], layer_tuples, theme: Theme) -> Legend | None:
    aes_scales: dict[str, Any] = group["aes"]
    aesthetics = tuple(aes_scales)
    breaks = group["breaks"]
    options: GuideLegend = group.get("options") or GuideLegend()
    mapped = {aes: list(scale.map(breaks)) for aes, scale in aes_scales.items()}

    contributing = []
    any_matched = False
    for geom, mapping, params in layer_tuples:
        matched = tuple(a for a in aesthetics if a in mapping)
        any_matched = any_matched or bool(matched)
        if _include_layer(_show_legend(params), matched, aesthetics):
            contributing.append((geom, matched if matched else aesthetics, params))
    if any_matched and not contributing:  # show_legend switched every layer off
        return None

    # linetype keys need room to tell the dash patterns apart
    key_theme = theme
    if "linetype" in aesthetics and getattr(theme, "legend_key_width", None) is None:
        key_theme = theme.replace(legend_key_width=_LINETYPE_KEY_WIDTH)

    fg = theme.get_foreground()
    keys = []
    for i in range(len(breaks)):
        override = {k: _override(v, i) for k, v in options.override_aes.items()}
        glyphs = VGroup()
        for geom, used, params in contributing:
            row = {a: mapped[a][i] for a in used}
            try:
                row.update(geom.aes_params(params))
            except AttributeError:
                pass
            row.update(override)
            try:
                glyph = geom.draw_key(row, key_theme)
            except NotImplementedError:
                continue
            if glyph is not None:
                glyphs.add(glyph.move_to(ORIGIN))
        if len(glyphs) == 0:  # no geom draws a key: plain filled square
            values = {a: mapped[a][i] for a in ("fill", "color") if a in mapped}
            values.update({a: override[a] for a in ("fill", "color") if a in override})
            colour = values.get("fill", values.get("color", fg))
            glyphs.add(
                Square(side_length=KEY_SIZE, fill_color=colour, fill_opacity=1, stroke_width=0)
            )
        keys.append(glyphs)

    text_style = theme.text_style("legend_text")
    labels = [
        theme.make_text(render_label(lab, theme.text_engine), **text_style)
        for lab in group["labels"]
    ]
    title = group["title"]
    if title is not None:
        if isinstance(title, Mobject):
            title = title.copy()
        else:
            title = theme.make_text(str(title), **theme.text_style("legend_title"))
    return Legend(title, keys, labels, aesthetics, breaks=breaks, theme=theme, options=options)


# ---------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------


def place_legends(
    panel_mobject: Mobject, legends: Sequence[Any], theme: Theme | None = None
) -> VGroup:
    """Lay out ``legends`` and put them beside (or inside) ``panel_mobject``.

    Each legend's keys follow its guide's ``direction``, else
    ``theme.legend_direction``, else horizontal for ``"top"``/``"bottom"``
    positions and vertical otherwise. Several legends are stacked per
    ``theme.legend_box`` (default: side by side for top/bottom, else one above
    the other) with ``theme.legend_spacing`` between them.

    ``theme.legend_position`` ``"right"``/``"left"``/``"top"``/``"bottom"``: the
    box goes ``theme.legend_box_spacing`` outside that side of the panel, aligned
    along it by ``theme.legend_justification`` (centred by default). An ``(x, y)``
    position puts the box inside the panel's plotting area (``panel.axes`` when
    present) at those normalised coordinates, with the box's justification point
    there. ``"none"`` (or no legends): an empty ``VGroup``. Only the legends are
    moved; the panel is not touched.
    """
    theme = theme if theme is not None else theme_default()
    position = theme.legend_position
    if position == "none" or not legends:
        return VGroup()
    side_horizontal = position in ("top", "bottom")
    default_dir = getattr(theme, "legend_direction", None) or (
        "horizontal" if side_horizontal else "vertical"
    )
    for leg in legends:
        own = getattr(getattr(leg, "options", None), "direction", None)
        leg.arrange_rows(own or default_dir)
    box = getattr(theme, "legend_box", None) or ("horizontal" if side_horizontal else "vertical")
    spacing = float(getattr(theme, "legend_spacing", _LEGEND_SPACING))
    group = VGroup(*legends)
    if box == "horizontal":
        group.arrange(RIGHT, buff=spacing, aligned_edge=UP)
    else:
        group.arrange(DOWN, buff=spacing, aligned_edge=LEFT)
    jx, jy = theme.legend_just() if hasattr(theme, "legend_just") else (0.5, 0.5)
    if isinstance(position, tuple):
        area = getattr(panel_mobject, "axes", panel_mobject)
        x0, y0 = area.get_corner(DL)[:2]
        x1, y1 = area.get_corner(UR)[:2]
        tx, ty = x0 + position[0] * (x1 - x0), y0 + position[1] * (y1 - y0)
        gx0, gy0 = group.get_corner(DL)[:2]
        group.shift((tx - (gx0 + jx * group.width), ty - (gy0 + jy * group.height), 0.0))
        return group
    buff = float(getattr(theme, "legend_box_spacing", _PANEL_BUFF))
    group.next_to(panel_mobject, _DIRECTIONS[position], buff=buff)
    gx0, gy0 = group.get_corner(DL)[:2]
    px0, py0 = panel_mobject.get_corner(DL)[:2]
    px1, py1 = panel_mobject.get_corner(UR)[:2]
    if side_horizontal:
        group.shift(((px0 + jx * (px1 - px0)) - (gx0 + jx * group.width), 0.0, 0.0))
    else:
        group.shift((0.0, (py0 + jy * (py1 - py0)) - (gy0 + jy * group.height), 0.0))
    return group


# ---------------------------------------------------------------------------
# component
# ---------------------------------------------------------------------------


class Guides:
    """Component holding ``aesthetic -> guide`` (``"legend"`` / ``"colourbar"`` /
    ``"none"`` / :func:`guide_legend` / :func:`guide_colorbar`)."""

    __slots__ = ("guides",)

    def __init__(self, **kw):
        out = {}
        for k, v in kw.items():
            if v is None or v is False:
                v = "none"
            if not isinstance(v, GUIDE_CLASSES) and v not in GUIDE_TYPES:
                raise ValueError(f"guide for {k!r} must be one of {GUIDE_TYPES}, got {v!r}")
            out[standardise_aes_name(k)] = v
        self.guides: dict[str, Any] = out

    def add_to(self, plot):
        return plot._with_guides(self.guides)

    def __repr__(self):
        return f"guides({', '.join(f'{k}={v!r}' for k, v in self.guides.items())})"


@register
def guides(**kw) -> Guides:
    """Set guides per aesthetic: ``guides(color="none")`` hides the colour legend;
    ``"legend"`` restores it (``None``/``False`` mean ``"none"``);
    ``guides(color=guide_legend(ncol=2), fill=guide_colorbar(reverse=True))``
    configures them."""
    return Guides(**kw)
