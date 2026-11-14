"""Batched marks: the fast path for large point / rect / tile / bar / segment layers.

Drawing one manim mobject per data row costs a deep copy, colour arrays and a
family walk per row, which dominates the build of a plot with thousands of marks
(see ``benchmarks/PROFILE.md``). In *fast* mode a geom instead draws one
``VMobject`` per distinct style -- ``(shape, colour, fill, alpha, stroke)`` for
points, ``(fill, alpha, outline, linewidth)`` for rectangles and bars,
``(colour, linewidth, alpha)`` for segments -- with every mark of that style a
subpath (a circle is the 8 cubic Bezier curves of manim's ``Circle``; a
rectangle its 4 straight edges), built straight from numpy arrays.

When fast mode is used
    The geoms take ``fast=None | True | False``. ``None`` (the default) batches
    a panel's layer when it has more than ``theme.fast_threshold`` rows (2000;
    ``None`` disables the switch) and does *not* map the ``key`` aesthetic
    (mapping ``key`` asks for per-mark identity). ``True`` always batches
    (warning that ``key`` is ignored if it is mapped); ``False`` never does.

What changes
    * ``layer.marks`` lists the batch mobjects (plus any marks drawn
      individually, below); ``layer.keys`` is ``None`` and ``layer.batched`` is
      ``True``. Per-mark animation, :func:`ggmanim.animate.mark` and key-based
      transitions are therefore not available (``mark`` raises ``TypeError``);
      whole-layer reveals (``fade_in``, ``create``, ``grow_from_baseline``)
      still work.
    * Draw order. Batches are drawn one after the other, and cairo fills all
      subpaths of a batch in one go (nonzero winding), so overlapping
      translucent marks of one batch would not darken each other, outlines
      would show through the fills of later marks, and where marks of
      different styles overlap the wrong one could end up on top. To keep the
      slow path's look, marks are split into *stacking layers*
      (:func:`stack_layers`): a mark goes above every earlier mark whose
      bounding box it overlaps -- of another style, or of its own style when
      that style is translucent or outlined in another colour -- and batches
      are emitted layer by layer, then style by style (in order of first
      appearance). A single opaque style is therefore always one batch, a few
      colours over a dense cloud need a few hundred. Overlap is tested on a
      raster of the boxes, so slivers thinner than about a quarter of a mark
      (e.g. a stroke overlapping the next tile) are not ordered: outlines of
      abutting tiles/bars and anti-aliased edges can differ by a few levels.

Marks that cannot be batched are drawn by the geom's normal per-row code and
added after the batches: points cut by the panel edge (they need
``clip_marker``), dashed rectangle outlines, dashed segments, and segments with
``arrow=`` (the whole layer is then drawn per row; ``fast=True`` warns).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from manim import ORIGIN, ManimColor, VGroup, VMobject

from ..scales.shape import parse_linetype, parse_shape
from ..theme import Theme
from ._clip import clip_segments, panel_rect
from ._marker_clip import marker_extent, near_box, scene_box
from ._markers import _prototype, make_marker, outline_width
from ._rect import clipped_rects, seal_seams, style_filled
from .base import color_resolver, finite_rows, num, num_array, resolve_color, transform_points

__all__ = [
    "DEFAULT_FAST_THRESHOLD",
    "use_fast",
    "batched_group",
    "is_batched",
    "stack_layers",
    "batch_mobject",
    "draw_points_fast",
    "draw_rects_fast",
    "draw_bars_fast",
    "draw_segments_fast",
]

#: Default for :attr:`ggmanim.theme.Theme.fast_threshold`.
DEFAULT_FAST_THRESHOLD = 2000
#: Scene units per manim ``stroke_width`` unit (cairo's ``cairo_line_width_multiple``).
_STROKE_UNIT = 0.01
#: Cells along the longer side of the :func:`stack_layers` raster, at most.
_MAX_RASTER = 2048

SlowDraw = Callable[[pd.DataFrame], VMobject]


# -- dispatch -------------------------------------------------------------------------


def use_fast(fast: bool | None, data: pd.DataFrame, theme: Theme | None, who: str) -> bool:
    """Whether a geom should draw ``data`` batched.

    ``fast=None``: more rows than ``theme.fast_threshold`` (``None`` = never) and
    no mapped ``key``. ``True``/``False`` force it (``True`` warns when ``key``
    is mapped, since batched marks carry no keys)."""
    keyed = "key" in data.columns
    if fast is None:
        threshold = getattr(theme, "fast_threshold", DEFAULT_FAST_THRESHOLD)
        return threshold is not None and len(data) > threshold and not keyed
    if fast and keyed:
        warnings.warn(
            f"{who}: fast=True draws batched marks; the 'key' aesthetic is ignored "
            "(no per-mark animation or mark() lookup)",
            stacklevel=4,
        )
    return bool(fast)


def batched_group(marks: Iterable[VMobject]) -> VGroup:
    """A geom's output for batched marks: ``VGroup(*marks)`` with ``.marks`` the
    batches, ``.keys = None`` and ``.batched = True`` (which
    :func:`ggmanim.animate.mark` checks)."""
    marks = list(marks)
    out: Any = VGroup(*marks)
    out.marks = marks
    out.keys = None
    out.batched = True
    return out


def is_batched(mob: Any) -> bool:
    """Whether ``mob`` (a geom's output) holds batched marks."""
    return bool(getattr(mob, "batched", False))


def _slow_marks(slow: SlowDraw, rows: pd.DataFrame) -> list[VMobject]:
    if len(rows) == 0:
        return []
    mob = slow(rows)
    marks = getattr(mob, "marks", None)
    return list(marks if marks is not None else mob.submobjects)


# -- grouping and stacking ----------------------------------------------------------


def _hex(c: Any) -> str | None:
    return c.to_hex() if isinstance(c, ManimColor) else None


def _hexer() -> Callable[[Any], str | None]:
    """:func:`_hex` memoised by object identity (resolved colours repeat as the
    same objects, see :func:`~.base.color_resolver`)."""
    cache: dict[int, tuple[Any, str | None]] = {}

    def hexof(c: Any) -> str | None:
        hit = cache.get(id(c))
        if hit is None or hit[0] is not c:
            hit = cache[id(c)] = (c, _hex(c))
        return hit[1]

    return hexof


def _groups(keys: Sequence[Hashable]) -> dict[Hashable, np.ndarray]:
    """``{key: row positions}`` in order of first appearance."""
    out: dict[Hashable, list[int]] = {}
    for i, k in enumerate(keys):
        out.setdefault(k, []).append(i)
    return {k: np.asarray(v, dtype=int) for k, v in out.items()}


def stack_layers(
    lo: np.ndarray,
    hi: np.ndarray,
    style: np.ndarray | None = None,
    self_overlap: np.ndarray | None = None,
) -> np.ndarray:
    """Stacking layer of each box (``lo``/``hi``: ``(n, 2)`` lower-left and
    upper-right corners, in draw order), such that a box is in a higher layer
    than every earlier box it overlaps. Drawing layer by layer then keeps every
    overlapping pair in data order.

    With ``style`` (an integer style id per box) and ``self_overlap`` (a bool per
    style id), overlaps between two boxes of the *same* style only count when
    ``self_overlap`` is true for it (for opaque, unoutlined marks the order
    within a style does not show); such a box may then share the layer of the
    earlier boxes of its style.

    Overlap is tested on a raster of the boxes (cells about a quarter of the
    median box's shorter side; a box covers the cells whose centres it contains, at least the
    one under its centre), so boxes that merely touch, like abutting tiles, do
    not overlap, and slivers thinner than a cell may be missed. Each box costs a
    few numpy operations on its cells.
    """
    lo = np.asarray(lo, dtype=float).reshape(-1, 2)
    hi = np.asarray(hi, dtype=float).reshape(-1, 2)
    n = len(lo)
    layer = np.zeros(n, dtype=np.int64)
    if n < 2:
        return layer
    sid = np.zeros(n, dtype=np.int64) if style is None else np.asarray(style, dtype=np.int64)
    same = (
        np.ones(int(sid.max()) + 1, dtype=bool)
        if self_overlap is None
        else np.asarray(self_overlap, dtype=bool)
    )
    origin = lo.min(axis=0)
    span = hi.max(axis=0) - origin
    ext = np.maximum(hi - lo, 0.0).min(axis=1)
    cell = max(float(np.median(ext)) / 4.0, float(span.max()) / _MAX_RASTER, 1e-12)
    shape = np.floor(span / cell).astype(np.int64) + 1
    a = np.ceil((lo - origin) / cell - 0.5).astype(np.int64)
    b = np.floor((hi - origin) / cell - 0.5).astype(np.int64)
    mid = np.floor(((lo + hi) / 2 - origin) / cell).astype(np.int64)
    empty = b < a  # thinner than a cell along that axis: the cell under the centre
    a[empty], b[empty] = mid[empty], mid[empty]
    a = np.clip(a, 0, shape - 1)
    b = np.clip(b, 0, shape - 1) + 1
    top = np.full((int(shape[1]), int(shape[0])), -1, dtype=np.int64)  # highest layer per cell
    who = np.full(top.shape, -1, dtype=np.int64)  # its style, -1 when several styles share it
    for i in range(n):
        ys, xs = slice(a[i, 1], b[i, 1]), slice(a[i, 0], b[i, 0])
        t, w = top[ys, xs], who[ys, xs]
        s = sid[i]
        m = int(t.max())
        if m < 0:
            k = 0
        elif same[s]:
            k = m + 1
        else:  # only the earlier boxes of other styles must stay below
            k = m + 1 if bool(((t == m) & (w != s)).any()) else m
        layer[i] = k
        w[(t == k) & (w != s)] = -1
        below = t < k
        w[below] = s
        t[below] = k
    return layer


def _needs_layers(style: VMobject) -> bool:
    """Would batching marks of ``style`` change how overlapping marks look?
    (translucent fill or stroke, or an outline in another colour than the fill)"""
    fill = float(style.get_fill_opacity())
    stroke = float(style.get_stroke_opacity()) if style.get_stroke_width() > 0 else 0.0
    if 0 < fill < 1 or 0 < stroke < 1:
        return True
    return fill > 0 and stroke > 0 and style.get_stroke_color() != style.get_fill_color()


def batch_mobject(points: np.ndarray, style: VMobject) -> VMobject:
    """One ``VMobject`` with Bezier ``points`` (``(4k, 3)``; subpaths split where
    consecutive curves do not meet) styled like ``style`` (fill, stroke, joint
    and cap style)."""
    m = VMobject()
    if len(points):
        m.set_points(np.asarray(points, dtype=float))
    stroke: Any = style.get_stroke_color()
    m.set_fill(style.get_fill_color(), opacity=style.get_fill_opacity())
    m.set_stroke(
        stroke,
        width=style.get_stroke_width(),
        opacity=style.get_stroke_opacity(),
    )
    m.joint_type = style.joint_type
    m.cap_style = style.cap_style
    return m


def _assemble(
    keys: Sequence[Hashable],
    style_of: Callable[[Hashable, int], VMobject],
    points_of: Callable[[np.ndarray], np.ndarray],
    lo: np.ndarray,
    hi: np.ndarray,
    pad_of: Callable[[VMobject], float] | None = None,
    after: Callable[[VMobject], VMobject] | None = None,
) -> list[VMobject]:
    """The batch mobjects for rows with style ``keys`` (``None``: not drawn here).

    ``style_of(key, first_row)`` styles a batch, ``points_of(rows)`` gives the
    ``(m, k, 3)`` Bezier points of those rows' marks (one style, so one ``k``),
    ``lo``/``hi`` the rows' bounding boxes (grown by ``pad_of(style)``, e.g. half
    the stroke). Batches are emitted by stacking layer (:func:`stack_layers`:
    marks of different styles that overlap keep their data order, as do
    overlapping marks of one style when :func:`_needs_layers`), then by style in
    order of first appearance; ``after`` post-processes each batch."""
    groups = [(k, idx) for k, idx in _groups(keys).items() if k is not None]
    if not groups:
        return []
    styles = [style_of(k, int(idx[0])) for k, idx in groups]
    n = len(keys)
    sid = np.full(n, -1, dtype=int)
    for s, (_k, idx) in enumerate(groups):
        sid[idx] = s
    rows = np.flatnonzero(sid >= 0)
    needs = np.array([_needs_layers(st) for st in styles], dtype=bool)
    layer = np.zeros(n, dtype=int)
    if len(groups) > 1 or needs.any():
        pad = np.array([pad_of(st) if pad_of is not None else 0.0 for st in styles])
        grow = pad[sid[rows]][:, None]
        layer[rows] = stack_layers(lo[rows] - grow, hi[rows] + grow, sid[rows], needs)
    order = rows[np.lexsort((sid[rows], layer[rows]))]  # by layer, then style
    cut = np.flatnonzero(np.diff(layer[order]) | np.diff(sid[order])) + 1
    out: list[VMobject] = []
    for part in np.split(order, cut):
        part = np.sort(part)
        m = batch_mobject(points_of(part).reshape(-1, 3), styles[sid[part[0]]])
        out.append(after(m) if after is not None else m)
    return out


def _half_stroke(style: VMobject) -> float:
    width = float(style.get_stroke_width())
    return width * _STROKE_UNIT / 2 if width > 0 else 0.0


# -- geoms ----------------------------------------------------------------------------


def _unit_points(base: str) -> np.ndarray:
    """Bezier points of the unit marker ``base`` (every subpath of its family)."""
    leaves = _prototype(base).family_members_with_points()
    return np.concatenate([np.asarray(m.points, dtype=float) for m in leaves])


def draw_points_fast(
    data: pd.DataFrame, panel_params: Any, coord: Any, theme: Theme, slow: SlowDraw
) -> VGroup:
    """Batched :class:`~ggmanim.geoms.point.GeomPoint` drawing. ``data`` has the
    default aesthetics filled in; ``slow(rows)`` draws rows one marker each
    (used for markers cut by the panel edge)."""
    data = finite_rows(data, ("x", "y"))
    pts = transform_points(coord, data["x"].to_numpy(float), data["y"].to_numpy(float))
    radius = num_array(data["size"], 0.06)
    rect = panel_rect(coord, panel_params)
    edge = np.zeros(len(pts), dtype=bool)
    if rect is not None:  # clip="on": drop far-away markers, clip the edge ones per row
        box = scene_box(coord, rect)
        ext = marker_extent(radius)
        keep = near_box(pts, box, ext)
        data, pts, radius = data[keep], pts[keep], radius[keep]
        edge = ~near_box(pts, box, -ext[keep])
    fg = theme.get_foreground()
    color_of, fill_of = color_resolver(fg), color_resolver()
    alpha = num_array(data["alpha"], 1.0)
    colors = [color_of(c) for c in data["color"].tolist()]
    fills = [fill_of(f) for f in data["fill"].tolist()]
    shapes = data["shape"].tolist()
    parsed: dict[str, tuple[str, str]] = {}
    hexof = _hexer()
    keys: list[Hashable] = []
    for s, c, f, a, r, e in zip(shapes, colors, fills, alpha, radius, edge):
        if e:
            keys.append(None)
            continue
        name = repr(s)
        if name not in parsed:
            parsed[name] = parse_shape(s)
        _base, style = parsed[name]
        sw = None if style == "solid" else outline_width(max(float(r), 0.0))
        keys.append((name, hexof(c), hexof(f), float(a), sw))
    reach = marker_extent(radius)[:, None]
    radius = np.maximum(radius, 0.0)

    def style_of(key: Hashable, i: int) -> VMobject:
        color = colors[i]
        marker = make_marker(
            shapes[i],
            ORIGIN,
            radius=float(radius[i]),
            color=color if color is not None else fg,
            fill=fills[i],
            alpha=float(alpha[i]),
        )
        leaf: Any = marker.family_members_with_points()[0]
        return leaf

    def points_of(idx: np.ndarray) -> np.ndarray:
        unit = _unit_points(parsed[repr(shapes[int(idx[0])])][0])
        r = np.where(radius[idx] > 0, radius[idx], 1e-6)
        return pts[idx][:, None, :] + r[:, None, None] * unit[None, :, :]

    out = _assemble(keys, style_of, points_of, pts[:, :2] - reach, pts[:, :2] + reach)
    out.extend(_slow_marks(slow, data[edge]))
    return batched_group(out)


def _solid(linetype: Any) -> bool:
    try:
        return parse_linetype(linetype) is None
    except ValueError:
        return False


def _ring_points(corners: np.ndarray) -> np.ndarray:
    """:func:`~._rect.ring_points` for every ring of ``corners`` (``(m, k, 3)``)
    at once: ``(m, 4k, 3)``."""
    a = corners
    b = np.roll(corners, -1, axis=1)
    out = np.stack([a, a + (b - a) / 3.0, a + 2.0 * (b - a) / 3.0, b], axis=2)
    return out.reshape(len(corners), -1, 3)


def _rect_batches(
    data: pd.DataFrame,
    corners: np.ndarray,
    fg: Any,
    style_of: Callable[[dict], VMobject],
    after: Callable[[VMobject], VMobject] | None,
) -> list[VMobject]:
    """Batches of the rectangles ``corners`` (rows of ``data``), one style per
    ``(fill, alpha, outline, linewidth)``; ``style_of(row)`` styles a batch."""
    if len(data) == 0:
        return []
    fill_of, line_of, hexof = color_resolver(fg), color_resolver(), _hexer()
    fills = [hexof(fill_of(v)) for v in data["fill"].tolist()]
    lines = [line_of(v) for v in data["color"].tolist()]
    alpha = num_array(data["alpha"], 1.0)
    width = num_array(data["linewidth"], 1.0)
    keys: list[Hashable] = [
        (f, float(al), hexof(ln), float(w) if ln is not None else None)
        for f, al, ln, w in zip(fills, alpha, lines, width)
    ]
    xy = corners[:, :, :2]
    return _assemble(
        keys,
        lambda _key, i: style_of(data.iloc[i].to_dict()),
        lambda idx: _ring_points(corners[idx]),
        xy.min(axis=1),
        xy.max(axis=1),
        None,  # outlines overlapping a neighbour by a stroke width are not ordered
        after,
    )


def draw_rects_fast(
    data: pd.DataFrame,
    panel_params: Any,
    coord: Any,
    theme: Theme,
    slow: SlowDraw,
    seal: bool = False,
) -> VGroup:
    """Batched :class:`~ggmanim.geoms.rect.GeomRect` / ``GeomTile`` drawing
    (``data``: rectangle columns and default aesthetics filled in). Rows with a
    dashed outline go through ``slow``; ``seal`` seals opaque batches like
    :func:`~._rect.seal_seams` does per tile."""
    line_of = color_resolver()
    dashed = np.array(
        [
            line_of(c) is not None and not _solid(lt)
            for c, lt in zip(data["color"].tolist(), data["linetype"].tolist())
        ],
        dtype=bool,
    )
    plain, corners = clipped_rects(data[~dashed], coord, panel_params)

    def style_of(row: dict) -> VMobject:
        return style_filled(VMobject(), row, theme)

    out = _rect_batches(
        plain, corners, theme.get_foreground(), style_of, seal_seams if seal else None
    )
    out.extend(_slow_marks(slow, data[dashed]))
    return batched_group(out)


def draw_bars_fast(data: pd.DataFrame, panel_params: Any, coord: Any, theme: Theme) -> VGroup:
    """Batched :class:`~ggmanim.geoms.col.GeomCol` drawing (``data``: bar
    extents and default aesthetics filled in). Styled like ``GeomCol._bar``:
    ``alpha`` applies to the fill and the outline."""
    fg = theme.get_foreground()
    data, corners = clipped_rects(data, coord, panel_params)

    def style_of(row: dict) -> VMobject:
        alpha = num(row.get("alpha"), 1.0)
        m = VMobject()
        m.set_fill(resolve_color(row.get("fill"), fg), opacity=alpha)
        line = resolve_color(row.get("color"))
        if line is None:
            m.set_stroke(width=0, opacity=0)
        else:
            m.set_stroke(color=line, width=num(row.get("linewidth"), 1.0), opacity=alpha)
        return m

    return batched_group(_rect_batches(data, corners, fg, style_of, None))


def draw_segments_fast(
    data: pd.DataFrame, panel_params: Any, coord: Any, theme: Theme, slow: SlowDraw
) -> VGroup:
    """Batched :class:`~ggmanim.geoms.segment.GeomSegment` drawing without
    arrowheads (``data``: default aesthetics filled in). Dashed rows go through
    ``slow``."""
    fg = theme.get_foreground()
    dashed = np.array([not _solid(lt) for lt in data["linetype"]], dtype=bool)
    rest, data = data[dashed], data[~dashed]
    x0, y0 = data["x"].to_numpy(float), data["y"].to_numpy(float)
    x1, y1 = data["xend"].to_numpy(float), data["yend"].to_numpy(float)
    rect = panel_rect(coord, panel_params)
    if rect is None:
        t0, t1 = np.zeros(len(x0)), np.ones(len(x0))
        vis = np.ones(len(x0), dtype=bool)
    else:
        t0, t1, vis = clip_segments(x0, y0, x1, y1, rect)
    a = transform_points(coord, x0 + t0 * (x1 - x0), y0 + t0 * (y1 - y0))
    b = transform_points(coord, x0 + t1 * (x1 - x0), y0 + t1 * (y1 - y0))
    if len(a):
        same = np.isclose(a, b, rtol=1e-05, atol=1e-08).all(axis=1)  # np.allclose per row
        vis = vis & ~same
    color_of = color_resolver(fg)
    colors = [color_of(c) for c in data["color"].tolist()]
    widths = num_array(data["linewidth"], 2.0)
    alphas = num_array(data["alpha"], 1.0)
    hexof = _hexer()
    keys: list[Hashable] = [
        (hexof(c), float(w), float(al)) if v else None
        for c, w, al, v in zip(colors, widths, alphas, vis)
    ]

    def style_of(_key: Hashable, i: int) -> VMobject:
        style = VMobject()
        style.set_fill(opacity=0)
        style.set_stroke(colors[i], width=float(widths[i]), opacity=float(alphas[i]))
        return style

    def points_of(idx: np.ndarray) -> np.ndarray:
        p, q = a[idx], b[idx]
        return np.stack([p, p + (q - p) / 3.0, p + 2.0 * (q - p) / 3.0, q], axis=1)

    lo, hi = np.minimum(a, b)[:, :2], np.maximum(a, b)[:, :2]
    out = _assemble(keys, style_of, points_of, lo, hi, _half_stroke)
    out.extend(_slow_marks(slow, rest))
    return batched_group(out)
