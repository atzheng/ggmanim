"""Data-keyed transitions between plots (gganimate's ``transition_*`` for manim).

::

    from ggmanim.transition import animate_reveal, animate_states, transition

    t = transition(p1952, p2007, key="country", limits="union")
    self.play(t)                 # points glide to their new place, axes rescale
    plot = t.plot                # the built second plot, now in the scene

    self.play(animate_states(spec, "year", key="country", title="Year {state}"))
    self.play(animate_reveal(spec, along="year", steps=12))

How a transition is put together
--------------------------------
Both plots are built with one layout: the second plot gets the first one's
panel size (``theme.panel_width``/``panel_height``) and legend position and is
then scaled and shifted so the two panels coincide. ``limits=`` decides whether
the position scales are shared (``"union"`` of both, ``"a"``, ``"b"``, or an
explicit ``{"x": (lo, hi), "y": (lo, hi)}`` in data space; continuous scales
only, discrete and date axes keep their own).

Then, part by part:

* **marks** -- per layer present in both plots (by name, or by position with
  ``match_layers="by_position"``), marks are matched by their ``key`` (the
  ``key`` aesthetic, set with ``key=`` or ``aes(key=...)``; otherwise the data
  row index / group id). Matched marks ``Transform``; marks only in the first
  plot fade out, marks only in the second fade in. Layers in only one plot fade.
* **axes** -- the axis lines transform. Tick labels, tick marks and major grid
  lines are matched by label text: shared ones move to their new place, the
  others fade, sliding along with the axis (an outgoing ``"20"`` moves to where
  20 is on the new axis) when both axes are continuous with the same transform.
  Minor grid lines match by value.
* **legend, titles, backgrounds** -- ``Transform`` when their shape is
  compatible (same text / same number of pieces), else a cross-fade.

The animation works on a copy of the first plot; when it finishes the scene
holds the second :class:`~ggmanim.mobjects.GGPlot` itself (``t.plot``), so
transitions chain: ``transition(t.plot, next_spec)``.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict, deque
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from manim import (
    RIGHT,
    UP,
    Animation,
    AnimationGroup,
    FadeIn,
    FadeOut,
    Mobject,
    Succession,
    Transform,
    Wait,
)
from manim.utils.rate_functions import linear

from .aes import Aes
from .scales.continuous import ScaleContinuousPosition
from .scales.date import ScaleDatePosition
from .scales.discrete_position import ScaleDiscretePosition
from .theme import theme_default

__all__ = [
    "Transition",
    "StatesAnimation",
    "transition",
    "animate_states",
    "animate_reveal",
    "shared_limits",
    "align_panels",
]

_LIMIT_MODES = ("union", "a", "b")


# -- specs ------------------------------------------------------------------------


def _is_spec(p: Any) -> bool:
    return hasattr(p, "build") and not isinstance(p, Mobject)


def _has_col(data: Any, col: str) -> bool:
    return isinstance(data, pd.DataFrame) and col in data.columns


def _with_key(spec: Any, key: str | None) -> Any:
    """``spec`` with ``aes(key=key)`` added to the plot mapping (when the plot
    data has the column) and to layers with their own data holding it; existing
    ``key`` mappings are kept."""
    if key is None:
        return spec
    if _has_col(spec.data, key) and "key" not in spec.mapping:
        spec = spec._replace(mapping=Aes({**dict(spec.mapping), "key": key}))
    layers = []
    for layer in spec.layers:
        mapping = layer.mapping if layer.mapping is not None else Aes()
        if _has_col(layer.data, key) and "key" not in mapping:
            layer = layer.replace(mapping=Aes({**dict(mapping), "key": key}))
        layers.append(layer)
    return spec._replace(layers=tuple(layers))


def _limitable(scale: Any) -> bool:
    return isinstance(scale, ScaleContinuousPosition) and not isinstance(scale, ScaleDatePosition)


def _position_limits(plot: Any) -> dict[str, tuple[float, float]]:
    """Data-space ``{"x": (lo, hi), "y": ...}`` of a built plot's continuous
    (non-date) position scales."""
    out: dict[str, tuple[float, float]] = {}
    for ax in ("x", "y"):
        scale = plot.scales.get(ax)
        if not _limitable(scale):
            continue
        lim = scale.get_limits()
        if lim is None:
            continue
        d = np.asarray(scale.trans.inverse(np.asarray(lim, dtype=float)), dtype=float)
        if np.all(np.isfinite(d)):
            out[ax] = (float(d.min()), float(d.max()))
    return out


def shared_limits(*plots: Any) -> dict[str, tuple[float, float]]:
    """The union of the continuous position-scale limits (data space) of
    ``plots`` (specs are built), as ``{"x": (lo, hi), "y": (lo, hi)}``."""
    out: dict[str, tuple[float, float]] = {}
    for p in plots:
        built = p.build() if _is_spec(p) else p
        for ax, (lo, hi) in _position_limits(built).items():
            if ax in out:
                lo, hi = min(lo, out[ax][0]), max(hi, out[ax][1])
            out[ax] = (lo, hi)
    return out


def _apply_limits(spec: Any, limits: Mapping[str, Any]) -> Any:
    """``spec`` with its continuous x/y scales' ``limits`` set (a copy of the
    spec's own scale when it has one, else ``xlim``/``ylim``). Discrete and date
    scales are left alone."""
    import copy

    from .scales.continuous import xlim, ylim

    for ax, lim in limits.items():
        if lim is None or ax not in ("x", "y"):
            continue
        lo, hi = (float(v) for v in lim)
        scale = spec.get_scale(ax)
        if scale is None:
            spec = spec.add((xlim if ax == "x" else ylim)(lo, hi))
        elif _limitable(scale):
            new = copy.copy(scale)
            new.limits = (lo, hi)
            spec = spec.add(new)
    return spec


def _panel_box(plot: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(lower-left, upper-right)`` scene corners of all the plot's panels."""
    pts = []
    for panel in plot.panels.values():
        ax = panel.axes
        pts.append(ax.c2p(ax.x_range[0], ax.y_range[0]))
        pts.append(ax.c2p(ax.x_range[1], ax.y_range[1]))
    arr = np.asarray(pts, dtype=float)
    return arr.min(axis=0), arr.max(axis=0)


def _build_like(spec: Any, ref: Any, ref_spec: Any = None) -> Any:
    """Build ``spec`` with the panel size of the built plot ``ref`` (and the
    legend position of ``ref_spec``'s theme when given), aligned onto it."""
    lo, hi = _panel_box(ref)
    theme = spec.theme if spec.theme is not None else theme_default()
    changes: dict[str, Any] = {
        "panel_width": float(hi[0] - lo[0]),
        "panel_height": float(hi[1] - lo[1]),
    }
    if ref_spec is not None:
        ref_theme = ref_spec.theme if ref_spec.theme is not None else theme_default()
        changes["legend_position"] = ref_theme.legend_position
    built = spec.build(theme=dataclasses.replace(theme, **changes))
    return align_panels(built, ref)


def align_panels(plot: Any, ref: Any) -> Any:
    """Scale and shift ``plot`` so its panel area coincides with ``ref``'s."""
    alo, ahi = _panel_box(ref)
    blo, bhi = _panel_box(plot)
    wb = bhi[0] - blo[0]
    if wb > 1e-9:
        f = (ahi[0] - alo[0]) / wb
        if abs(f - 1.0) > 1e-9:
            plot.scale(f, about_point=blo)
    plot.shift(alo - _panel_box(plot)[0])
    return plot


# -- matching ---------------------------------------------------------------------


def _has_points(m: Mobject | None) -> bool:
    return m is not None and len(m.family_members_with_points()) > 0


def _text_of(m: Mobject | None) -> str | None:
    if m is None:
        return None
    for attr in ("original_text", "text", "tex_string"):
        v = getattr(m, attr, None)
        if isinstance(v, str):
            return v
    return None


def _hkey(k: Any) -> Hashable:
    try:
        hash(k)
    except TypeError:
        return repr(k)
    return k


class _Builder:
    """Collects the animations of one transition and which mobjects of each side
    they cover (whatever is left over fades at the end)."""

    def __init__(self, a: Mobject, b: Mobject):
        self.a, self.b = a, b
        self.anims: list[Animation] = []
        self._seen_a: set[int] = set()
        self._seen_b: set[int] = set()

    def _mark(self, seen: set[int], m: Mobject) -> None:
        seen.update(id(x) for x in m.get_family())

    def transform(self, x: Mobject, y: Mobject, out: list | None = None) -> None:
        self._mark(self._seen_a, x)
        self._mark(self._seen_b, y)
        (self.anims if out is None else out).append(Transform(x, y))

    def fade_out(self, x: Mobject, shift: np.ndarray | None = None, out: list | None = None):
        self._mark(self._seen_a, x)
        if _has_points(x):
            kw = {"shift": shift} if shift is not None else {}
            (self.anims if out is None else out).append(FadeOut(x, **kw))

    def fade_in(self, y: Mobject, shift: np.ndarray | None = None, out: list | None = None):
        self._mark(self._seen_b, y)
        if _has_points(y):
            kw = {"shift": shift} if shift is not None else {}
            (self.anims if out is None else out).append(FadeIn(y, **kw))

    def pair(self, x: Mobject | None, y: Mobject | None, out: list | None = None) -> None:
        """Transform when both exist with compatible shape, else cross-fade."""
        hx, hy = _has_points(x), _has_points(y)
        if hx and hy:
            assert x is not None and y is not None
            tx, ty = _text_of(x), _text_of(y)
            same_text = tx is None and ty is None or tx == ty
            nx, ny = len(x.family_members_with_points()), len(y.family_members_with_points())
            if same_text and (nx == ny or (tx is None and ty is None and min(nx, ny) > 0)):
                self.transform(x, y, out)
                return
            d = y.get_center() - x.get_center()
            self.fade_out(x, d, out)
            self.fade_in(y, d, out)
            return
        if x is not None:
            self.fade_out(x, out=out)
        if y is not None:
            self.fade_in(y, out=out)

    def leftovers(self) -> list[Animation]:
        out: list[Animation] = []
        for m in self.a.family_members_with_points():
            if id(m) not in self._seen_a:
                out.append(FadeOut(m))
        for m in self.b.family_members_with_points():
            if id(m) not in self._seen_b:
                out.append(FadeIn(m))
        return out


# -- axes -------------------------------------------------------------------------


class _Tick:
    __slots__ = ("value", "label", "parts")

    def __init__(self, value: float, label: str):
        self.value = value
        self.label = label
        self.parts: dict[str, Mobject] = {}


def _axis_scale(plot: Any, cell: Any, horizontal: bool, flipped: bool) -> Any:
    aes = ("y" if flipped else "x") if horizontal else ("x" if flipped else "y")
    per_panel = getattr(plot, "panel_scales", None) or {}
    scales = per_panel.get(cell) or plot.scales
    return scales.get(aes)


def _comparable(sa: Any, sb: Any) -> bool:
    if not isinstance(sa, ScaleContinuousPosition) or not isinstance(sb, ScaleContinuousPosition):
        return False
    if isinstance(sa, ScaleDiscretePosition) or type(sa) is not type(sb):
        return False
    return getattr(sa.trans, "name", None) == getattr(sb.trans, "name", None)


def _pos(panel: Any, value: float, horizontal: bool) -> float:
    ax = panel.axes
    if horizontal:
        return float(ax.c2p(value, ax.y_range[0])[0])
    return float(ax.c2p(ax.x_range[0], value)[1])


def _line_value(panel: Any, line: Mobject, horizontal: bool) -> float | None:
    """The axis value of a grid line perpendicular to the ``horizontal`` axis
    (a vertical line for the horizontal axis), else ``None``."""
    try:
        start, end = line.get_start(), line.get_end()
    except Exception:  # noqa: BLE001 - not a line
        return None
    d = end - start
    vertical = abs(d[0]) < abs(d[1])
    if vertical != horizontal:
        return None
    coords = panel.axes.p2c(start)
    return float(coords[0] if horizontal else coords[1])


def _ticks(panel: Any, horizontal: bool) -> tuple[list[_Tick], list[tuple[float, Mobject]]]:
    """Tick items (value, label, {tick, label, grid}) and minor grid lines
    ``[(value, line)]`` of one axis of a panel."""
    pp = panel.panel_params
    flipped = bool(getattr(panel, "flipped", False))
    use_x = horizontal != flipped
    breaks = pp.x_breaks if use_x else pp.y_breaks
    labels = pp.x_labels if use_x else pp.y_labels
    items = [_Tick(float(v), str(s)) for v, s in zip(breaks, labels)]
    ticks = panel.x_ticks if horizontal else panel.y_ticks
    texts = panel.x_tick_labels if horizontal else panel.y_tick_labels
    if len(ticks) == len(items):
        for it, m in zip(items, ticks):
            it.parts["tick"] = m
    if len(texts) == len(items):
        for it, m in zip(items, texts):
            it.parts["label"] = m
    span = abs(float(panel.axes.x_range[1] - panel.axes.x_range[0]))
    if not horizontal:
        span = abs(float(panel.axes.y_range[1] - panel.axes.y_range[0]))
    tol = 1e-6 * max(span, 1e-12)
    for line in panel.grid_major:
        v = _line_value(panel, line, horizontal)
        if v is None:
            continue
        for it in items:
            if "grid" not in it.parts and abs(it.value - v) <= tol:
                it.parts["grid"] = line
                break
    minor = []
    for line in panel.grid_minor:
        v = _line_value(panel, line, horizontal)
        if v is not None:
            minor.append((v, line))
    return items, minor


def _axis_anims(
    bld: _Builder, pa: Any, pb: Any, horizontal: bool, comparable: bool, out: list
) -> None:
    unit = RIGHT if horizontal else UP
    k = 0 if horizontal else 1
    ends = [pa.axes.get_critical_point(-unit)[k], pa.axes.get_critical_point(unit)[k]]

    def delta(v: float) -> np.ndarray | None:
        """Where ``v`` moves along the axis (its place on ``pb`` minus its place
        on ``pa``), each end clamped to the panel so nothing flies in from afar."""
        if not comparable:
            return None
        p_a, p_b = _pos(pa, v, horizontal), _pos(pb, v, horizontal)
        if not (np.isfinite(p_a) and np.isfinite(p_b)):
            return None
        d = float(np.clip(p_b, *ends) - np.clip(p_a, *ends))
        return d * unit

    items_a, minor_a = _ticks(pa, horizontal)
    items_b, minor_b = _ticks(pb, horizontal)
    by_label: dict[str, deque[_Tick]] = defaultdict(deque)
    for it in items_b:
        by_label[it.label].append(it)
    matched_b: set[int] = set()
    for it in items_a:
        queue = by_label.get(it.label)
        other = queue.popleft() if queue else None
        if other is None or it.label == "":
            for m in it.parts.values():
                bld.fade_out(m, delta(it.value), out)
            continue
        matched_b.add(id(other))
        for part in ("tick", "label", "grid"):
            bld.pair(it.parts.get(part), other.parts.get(part), out)
    for it in items_b:
        if id(it) in matched_b:
            continue
        d = delta(it.value)
        for m in it.parts.values():
            bld.fade_in(m, d, out)

    span = abs(float((pb.axes.x_range if horizontal else pb.axes.y_range)[1]))
    used: set[int] = set()
    for v, line in minor_a:
        hit = None
        for j, (w, _) in enumerate(minor_b):
            if j not in used and np.isclose(v, w, rtol=1e-9, atol=1e-9 * max(span, 1.0)):
                hit = j
                break
        if hit is None:
            bld.fade_out(line, delta(v), out)
        else:
            used.add(hit)
            bld.transform(line, minor_b[hit][1], out)
    for j, (w, line_b) in enumerate(minor_b):
        if j not in used:
            bld.fade_in(line_b, delta(w), out)


def _panel_anims(bld: _Builder, a: Any, b: Any, cell: Any, pa: Any, pb: Any) -> list:
    out: list[Animation] = []
    bld.pair(getattr(pa, "background", None), getattr(pb, "background", None), out)
    bld.transform(pa.axes, pb.axes, out)
    for horizontal in (True, False):
        flipped = bool(getattr(pa, "flipped", False))
        sa = _axis_scale(a, cell, horizontal, flipped)
        sb = _axis_scale(b, cell, horizontal, bool(getattr(pb, "flipped", False)))
        _axis_anims(bld, pa, pb, horizontal, _comparable(sa, sb), out)
    bld.pair(pa.x_title, pb.x_title, out)
    bld.pair(pa.y_title, pb.y_title, out)
    bld.pair(getattr(pa, "border", None), getattr(pb, "border", None), out)
    return out


# -- layers -----------------------------------------------------------------------


def _layer_pairs(a: Any, b: Any, match_layers: str) -> tuple[list, list, list]:
    ka, kb = list(a.layers), list(b.layers)
    if match_layers == "by_name":
        both = [(k, k) for k in ka if k in kb]
    elif match_layers == "by_position":
        both = list(zip(ka, kb))
    else:
        raise ValueError(f"match_layers must be 'by_name' or 'by_position', got {match_layers!r}")
    used_a = {k for k, _ in both}
    used_b = {k for _, k in both}
    return both, [k for k in ka if k not in used_a], [k for k in kb if k not in used_b]


def _mark_family(marks: Iterable[Mobject]) -> set[int]:
    return {id(x) for m in marks for x in m.get_family()}


def _layer_anims(bld: _Builder, ma: Mobject, mb: Mobject) -> tuple[list[Animation], dict[str, int]]:
    """Mark-by-mark animations of one layer and ``{matched, exited, entered}``."""
    out: list[Animation] = []
    marks_a, keys_a = getattr(ma, "marks", None), getattr(ma, "keys", None)
    marks_b, keys_b = getattr(mb, "marks", None), getattr(mb, "keys", None)
    batched = getattr(ma, "batched", False) or getattr(mb, "batched", False)
    if marks_a is None or keys_a is None or marks_b is None or keys_b is None or batched:
        bld.pair(ma, mb, out)
        return out, {"matched": 0, "exited": 0, "entered": 0}
    index: dict[Hashable, deque[int]] = defaultdict(deque)
    for j, k in enumerate(keys_b):
        index[_hkey(k)].append(j)
    stats = {"matched": 0, "exited": 0, "entered": 0}
    taken: set[int] = set()
    for m, k in zip(marks_a, keys_a):
        queue = index.get(_hkey(k))
        if queue:
            j = queue.popleft()
            taken.add(j)
            stats["matched"] += 1
            bld.pair(m, marks_b[j], out)
        else:
            stats["exited"] += 1
            bld.fade_out(m, out=out)
    for j, m in enumerate(marks_b):
        if j not in taken:
            stats["entered"] += 1
            bld.fade_in(m, out=out)
    # the layer's other pieces (e.g. direct labels): fade
    fam_a, fam_b = _mark_family(marks_a), _mark_family(marks_b)
    for m in ma.family_members_with_points():
        if id(m) not in fam_a:
            bld.fade_out(m, out=out)
    for m in mb.family_members_with_points():
        if id(m) not in fam_b:
            bld.fade_in(m, out=out)
    return out, stats


# -- the animation ----------------------------------------------------------------


def _drop_family(scene: Any, *mobs: Mobject) -> None:
    """Take ``mobs`` (and anything of their family added on its own) out of the
    scene, including the list of mobjects redrawn during the current ``play``."""
    ids = {id(x) for m in mobs for x in m.get_family()}
    scene.mobjects = [m for m in scene.mobjects if id(m) not in ids]
    moving = getattr(scene, "moving_mobjects", None)
    if moving:
        scene.moving_mobjects = [m for m in moving if id(m) not in ids]


def _in_scene(scene: Any, mob: Mobject) -> bool:
    return any(m is mob for top in scene.mobjects for m in top.get_family())


class Transition(AnimationGroup):
    """A transition from built plot ``plot_a`` to built plot ``plot_b``; see
    :func:`transition`.

    Attributes
    ----------
    plot_a, plot : the two :class:`~ggmanim.mobjects.GGPlot` (``plot`` is the
        second one; the scene holds it once the animation has finished)
    working : the copy of ``plot_a`` that is animated
    layer_animations : ``{layer key: [Animation]}`` the mark animations of each
        layer in both plots
    layer_stats : ``{layer key: {"matched", "exited", "entered"}}`` mark counts
    """

    def __init__(
        self,
        plot_a: Any,
        plot_b: Any,
        *,
        lag_ratio: float = 0.0,
        match_layers: str = "by_name",
        run_time: float = 1.5,
        **kwargs: Any,
    ):
        self.plot_a = plot_a
        self.plot = plot_b
        self.working = plot_a.copy()
        self.layer_animations: dict[Any, list[Animation]] = {}
        self.layer_stats: dict[Any, dict[str, int]] = {}
        anims = self._animations(self.working, plot_b, lag_ratio, match_layers)
        self._scene: Any = None
        self._swapped = False
        # an introducer, so Scene.play does not add the group of all parts itself;
        # _setup_scene puts the working copy in the scene instead
        kwargs.setdefault("introducer", True)
        super().__init__(*anims, run_time=run_time, **kwargs)

    def _animations(self, a: Any, b: Any, lag: float, match_layers: str) -> list[Animation]:
        bld = _Builder(a, b)
        bld.pair(a.background, b.background)
        pa, pb = a.panels, b.panels
        for cell in list(pa) + [c for c in pb if c not in pa]:
            if cell in pa and cell in pb:
                bld.anims.extend(_panel_anims(bld, a, b, cell, pa[cell], pb[cell]))
            elif cell in pa:
                bld.fade_out(pa[cell])
            else:
                bld.fade_in(pb[cell])
        bld.pair(a.strips, b.strips)
        ga, gb = a.panel_grid, b.panel_grid
        if ga is not None or gb is not None:
            for attr in ("x_title", "y_title"):
                bld.pair(getattr(ga, attr, None), getattr(gb, attr, None))
        both, only_a, only_b = _layer_pairs(a, b, match_layers)
        for ka, kb in both:
            anims, stats = _layer_anims(bld, a.layers[ka], b.layers[kb])
            self.layer_animations[kb] = anims
            self.layer_stats[kb] = stats
            if anims:
                bld.anims.append(AnimationGroup(*anims, lag_ratio=lag))
        for k in only_a:
            bld.fade_out(a.layers[k])
        for k in only_b:
            bld.fade_in(b.layers[k])
        bld.pair(a.legend, b.legend)
        for attr in ("title", "subtitle", "caption"):
            bld.pair(getattr(a, attr), getattr(b, attr))
        rest = bld.leftovers()
        return bld.anims + rest

    def _setup_scene(self, scene: Any) -> None:
        if scene is not None:
            self._scene = scene
            _drop_family(scene, self.plot_a, self.plot)
            scene.add(self.working)
        super()._setup_scene(scene)

    def begin(self) -> None:
        self._swapped = False
        super().begin()

    def finish(self) -> None:
        super().finish()
        scene = self._scene
        if scene is not None and not self._swapped:
            self._swapped = True
            _drop_family(scene, self.working, self.plot)
            scene.add(self.plot)


def transition(
    plot_a: Any,
    plot_b: Any,
    key: str | None = None,
    *,
    run_time: float = 1.5,
    lag_ratio: float = 0.0,
    match_layers: str = "by_name",
    limits: str | Mapping[str, Any] | None = None,
    align: bool = True,
    **kwargs: Any,
) -> Transition:
    """Morph one plot into another, mark by mark.

    Parameters
    ----------
    plot_a, plot_b : ``ggplot`` spec or built :class:`~ggmanim.mobjects.GGPlot`
        Specs are built here; ``plot_b`` gets ``plot_a``'s panel size (and, for a
        spec ``plot_a``, its legend position) and is aligned onto it.
    key : str, optional
        Column identifying the same thing in both plots (``aes(key=...)`` is
        added to the specs; built plots use the ``key`` they were built with).
        Without it marks match by data row index / group id.
    run_time : float
        Seconds.
    lag_ratio : float
        Stagger between the marks of a layer.
    match_layers : ``"by_name"`` or ``"by_position"``
        How layers pair up.
    limits : ``"union"``, ``"a"``, ``"b"``, mapping or ``None``
        Shared continuous position limits (data space): the union of both
        plots', one plot's, or ``{"x": (lo, hi), "y": (lo, hi)}``. ``None``:
        each plot keeps its own and the axes rescale. A built ``plot_a`` cannot
        be rebuilt, so then only ``plot_b`` gets the limits.
    align : bool
        Scale/shift a built ``plot_b`` onto ``plot_a``'s panel (specs always are).
    **kwargs
        Passed to the :class:`Transition` (``rate_func`` ...).

    Returns
    -------
    :class:`Transition`, whose ``.plot`` is the built second plot.
    """
    spec_a = _with_key(plot_a, key) if _is_spec(plot_a) else None
    spec_b = _with_key(plot_b, key) if _is_spec(plot_b) else None
    built_a = spec_a.build() if spec_a is not None else plot_a

    if limits is not None:
        if isinstance(limits, str):
            if limits not in _LIMIT_MODES:
                raise ValueError(f"limits must be one of {_LIMIT_MODES}, a mapping or None")
            if limits == "a":
                target = _position_limits(built_a)
            elif limits == "b":
                target = shared_limits(spec_b if spec_b is not None else plot_b)
            else:
                target = shared_limits(built_a, spec_b if spec_b is not None else plot_b)
        else:
            target = dict(limits)
        if spec_a is not None and target != _position_limits(built_a):
            spec_a = _apply_limits(spec_a, target)
            built_a = spec_a.build()
        if spec_b is not None:
            spec_b = _apply_limits(spec_b, target)

    if spec_b is not None:
        built_b = _build_like(spec_b, built_a, spec_a)
    else:
        built_b = align_panels(plot_b, built_a) if align else plot_b
    return Transition(
        built_a,
        built_b,
        lag_ratio=lag_ratio,
        match_layers=match_layers,
        run_time=run_time,
        **kwargs,
    )


# -- states and reveal -------------------------------------------------------------


class StatesAnimation(Succession):
    """A chain of :class:`Transition` with pauses; see :func:`animate_states`.

    ``plots`` are the built plots (one per state, plus the first again when
    wrapping), ``states`` the state values, ``transitions`` the transitions. The
    first plot is added to the scene when the animation starts (if it is not
    there already, e.g. after ``self.play(anim.plots[0].create())``).
    """

    def __init__(
        self,
        plots: list[Any],
        states: list[Any],
        transitions: list[Transition],
        anims: list[Animation],
        **kwargs: Any,
    ):
        self.plots = plots
        self.states = states
        self.transitions = transitions
        self.plot = plots[-1]
        kwargs.setdefault("introducer", True)
        super().__init__(*anims, **kwargs)
        # so the first plot is redrawn every frame (not painted into the static
        # background) and can be taken out by the first transition
        self.group.add(plots[0])

    def _setup_scene(self, scene: Any) -> None:
        # not Succession._setup_scene: as an introducer it would add the Waits'
        # placeholder mobjects; each Transition sets itself up when it starts
        if scene is None:
            return
        if not _in_scene(scene, self.plots[0]):
            scene.add(self.plots[0])
        self.scene = scene


def _state_values(col: pd.Series, levels: Sequence[Any] | None) -> list[Any]:
    if levels is not None:
        return list(levels)
    if isinstance(col.dtype, pd.CategoricalDtype):
        present = set(col.dropna().unique())
        return [c for c in col.cat.categories if c in present]
    vals = list(pd.unique(col.dropna()))
    try:
        return sorted(vals)
    except TypeError:
        return vals


def _map_data(spec: Any, col: str, fn: Any) -> Any:
    """``spec`` with ``fn(df)`` applied to the plot data and to every layer's own
    data frame that has column ``col``."""
    if _has_col(spec.data, col):
        spec = spec._replace(data=fn(spec.data))
    layers = tuple(
        layer.replace(data=fn(layer.data)) if _has_col(layer.data, col) else layer
        for layer in spec.layers
    )
    return spec._replace(layers=layers)


def _chain(
    specs: list[Any],
    states: list[Any],
    *,
    key: str | None,
    limits: str | Mapping[str, Any] | None,
    transition_length: float,
    state_length: float,
    wrap: bool,
    lag_ratio: float,
    **kwargs: Any,
) -> StatesAnimation:
    specs = [_with_key(s, key) for s in specs]
    if limits == "union":
        shared: Mapping[str, Any] | None = shared_limits(*specs)
    elif limits is None or isinstance(limits, Mapping):
        shared = limits
    else:
        raise ValueError("limits must be 'union', a mapping or None here")
    if shared:
        specs = [_apply_limits(s, shared) for s in specs]
    first = specs[0].build()
    plots = [first]
    transitions: list[Transition] = []
    anims: list[Animation] = []
    order = list(range(1, len(specs))) + ([0] if wrap and len(specs) > 1 else [])
    for i in order:
        if state_length > 0:
            anims.append(Wait(run_time=state_length))
        t = transition(
            plots[-1], specs[i], run_time=transition_length, lag_ratio=lag_ratio, **kwargs
        )
        transitions.append(t)
        plots.append(t.plot)
        anims.append(t)
    if state_length > 0:
        anims.append(Wait(run_time=state_length))
    if not anims:
        anims.append(Wait(run_time=1e-3))
    return StatesAnimation(plots, list(states), transitions, anims)


def animate_states(
    spec: Any,
    states: str,
    *,
    key: str | None = None,
    levels: Sequence[Any] | None = None,
    transition_length: float = 1.0,
    state_length: float = 0.5,
    wrap: bool = False,
    limits: str | Mapping[str, Any] | None = "union",
    title: str | None = None,
    lag_ratio: float = 0.0,
    **kwargs: Any,
) -> StatesAnimation:
    """gganimate's ``transition_states``: one plot per value of column
    ``states`` (the rows with that value), shown for ``state_length`` seconds
    and morphed into the next over ``transition_length``.

    Parameters
    ----------
    spec : ``ggplot`` spec; its data (and layers' own data with the column) is
        filtered per state
    states : column name
    key : column matching marks between states (see :func:`transition`)
    levels : state values in order (default: category order, else sorted)
    wrap : transition from the last state back to the first
    limits : shared position limits, ``"union"`` of all states (default), a
        mapping or ``None`` (axes rescale)
    title : format string for a per-state title, e.g. ``"Year: {state}"``
    lag_ratio, **kwargs : passed to each :func:`transition`
    """
    from .plot import labs

    data = spec.data
    if not _has_col(data, states):
        frames = [l.data for l in spec.layers if _has_col(l.data, states)]
        if not frames:
            raise ValueError(f"no data with a column {states!r}")
        data = frames[0]
    values = _state_values(data[states], levels)
    if not values:
        raise ValueError(f"column {states!r} has no values")
    specs = []
    for v in values:
        s = _map_data(spec, states, lambda df, v=v: df[df[states] == v])
        if title is not None:
            s = s.add(labs(title=title.format(state=v)))
        specs.append(s)
    return _chain(
        specs,
        values,
        key=key,
        limits=limits,
        transition_length=transition_length,
        state_length=state_length,
        wrap=wrap,
        lag_ratio=lag_ratio,
        **kwargs,
    )


def animate_reveal(
    spec: Any,
    along: str | None = None,
    *,
    steps: int = 10,
    run_time: float = 4.0,
    key: str | None = None,
    **kwargs: Any,
) -> StatesAnimation:
    """gganimate's ``transition_reveal``: draw the data progressively along
    column ``along`` (default: the column mapped to ``x``) in ``steps`` steps.

    Every state keeps all rows (rows beyond the step get a missing ``along``
    value and are dropped by the geoms), so groups, colours and legends stay
    stable; position limits are those of the full data. Points match by row,
    lines grow. Each step plays linearly; ``**kwargs`` go to each transition.
    """
    if along is None:
        mapped = spec.mapping.get("x") if "x" in spec.mapping else None
        if not isinstance(mapped, str):
            raise ValueError("along= is required when x is not mapped to a column")
        along = mapped
    data = spec.data
    if not _has_col(data, along):
        raise ValueError(f"the plot data has no column {along!r}")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    vals = np.sort(pd.unique(data[along].dropna()))
    if len(vals) == 0:
        raise ValueError(f"column {along!r} has no values")
    idx = np.unique(np.linspace(0, len(vals) - 1, steps + 1).round().astype(int))[1:]
    if len(idx) == 0:
        idx = np.array([len(vals) - 1])
    cuts = [vals[i] for i in idx]
    specs = [
        _map_data(
            spec, along, lambda df, t=t: df.assign(**{along: df[along].where(df[along] <= t)})
        )
        for t in cuts
    ]
    full = _with_key(spec, key)
    limits = _position_limits(full.build())
    kwargs.setdefault("rate_func", linear)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _chain(
            specs,
            cuts,
            key=key,
            limits=limits,
            transition_length=run_time / max(len(specs) - 1, 1),
            state_length=0.0,
            wrap=False,
            lag_ratio=0.0,
            **kwargs,
        )
