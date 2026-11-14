"""Animating built plots: ordered reveal, per-layer animations, mark lookup.

The usual entry points are the :class:`~ggmanim.mobjects.GGPlot` methods, which
call the functions here::

    plot = p.build()
    self.play(plot.create())                      # panel, axes, layers, legend, titles
    self.play(plot.hide_layer("fit"))
    self.play(plot.reveal_layer("fit", run_time=2))
    self.play(Indicate(plot.mark("points", "Ohio")))   # aes(key="state")

Per-geom animations (registry)
------------------------------
How a layer appears is looked up in :data:`ANIMATIONS`, a registry keyed by
``Geom`` class *name*. The lookup walks the geom's MRO, so a subclass inherits
its parent's entry (``GeomFunction`` -> ``GeomPath``) unless it registers its
own. New geoms register from their own module, without editing this file::

    from ggmanim.animate import register_animation
    register_animation("GeomViolin", "fade_in")               # a preset by name
    register_animation(GeomViolin, my_factory, hide=my_hide)  # or a factory

A factory is ``factory(mob, ctx) -> Animation | None``: ``mob`` is the layer's
mobject for *one* panel (``mob.marks`` / ``mob.keys`` set by
:func:`~ggmanim.geoms.base.attach_marks`) and ``ctx`` an
:class:`AnimationContext` (the panel, ``flipped``, ``mark_lag``, extra animation
keyword arguments). Presets (:data:`PRESETS`): ``"fade_in"``, ``"create"``,
``"write"``, ``"grow"`` (``GrowFromCenter`` per mark), ``"grow_from_baseline"``
(bars stretch out of the value-axis zero line; horizontal under
``coord_flip``) and ``"grow_from_edge"`` (from the bottom edge, the left edge
under ``coord_flip``). ``hide`` defaults to the reveal played backwards.

Which geom drew a layer
-----------------------
``GGPlot.layer_geoms`` (``{layer key: Geom}``) tells the lookup which geom drew
each layer; the build records it (:func:`attach_geoms` still sets it from a
spec, overriding the recorded geoms). Layers whose
geom is unknown get an animation inferred from their marks: text -> ``write``,
open strokes -> ``create``, anything else -> ``fade_in``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from manim import (
    Animation,
    AnimationGroup,
    Create,
    FadeIn,
    GrowFromCenter,
    Mobject,
    Transform,
    VMobject,
    Write,
)
from manim.utils.rate_functions import smooth

__all__ = [
    "ANIMATIONS",
    "DEFAULT_ORDER",
    "PRESETS",
    "AnimationContext",
    "AnimationEntry",
    "GrowFromBaseline",
    "animation_for",
    "attach_geoms",
    "create_plot",
    "hide_layer",
    "mark",
    "register_animation",
    "reveal_layer",
]

#: Default stage order of :func:`create_plot`.
DEFAULT_ORDER: tuple[str, ...] = ("panel", "axes", "layers", "legend", "titles")
_STAGES = frozenset(DEFAULT_ORDER)

Factory = Callable[[Any, "AnimationContext"], "Animation | None"]


@dataclass
class AnimationContext:
    """What a per-geom animation factory gets besides the layer mobject.

    Attributes
    ----------
    plot : the :class:`~ggmanim.mobjects.GGPlot`
    layer : the layer key
    geom : the ``Geom`` that drew the layer (``None`` if unknown)
    panel : the :class:`~ggmanim.coords.CartesianPanel` this part is drawn in
    flipped : ``coord_flip`` (bars grow horizontally)
    mark_lag : ``lag_ratio`` between marks
    kwargs : extra keyword arguments for the animation (``rate_func`` ...)
    """

    plot: Any
    layer: Any
    geom: Any
    panel: Any
    flipped: bool = False
    mark_lag: float = 0.05
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnimationEntry:
    """A registry entry: ``reveal``/``hide`` factories and an optional
    ``batched(mob) -> bool`` predicate (marks are batches, not data rows, so
    :func:`mark` refuses per-row lookup)."""

    reveal: Factory
    hide: Factory | None = None
    batched: Callable[[Mobject], bool] | None = None


# -- helpers ------------------------------------------------------------------------


def _has_points(mob: Mobject | None) -> bool:
    return mob is not None and len(mob.family_members_with_points()) > 0


def _marks(mob: Mobject) -> list[Mobject]:
    marks = getattr(mob, "marks", None)
    return list(marks) if marks is not None else list(mob.submobjects)


def _others(mob: Mobject) -> list[Mobject]:
    """Submobjects of ``mob`` that are not marks (e.g. line labels)."""
    marks = {id(m) for m in _marks(mob)}
    return [s for s in mob.submobjects if id(s) not in marks]


def _fade_others(mob: Mobject) -> list[Animation]:
    return [FadeIn(m) for m in _others(mob) if _has_points(m)]


def _group(anims: Iterable[Animation | None], **kwargs) -> Animation | None:
    kept: list[Animation] = [a for a in anims if a is not None]
    if not kept:
        return None
    if len(kept) == 1 and not kwargs:
        return kept[0]
    return AnimationGroup(*kept, **kwargs)


# -- presets ------------------------------------------------------------------------


def fade_in(mob: Any, ctx: AnimationContext) -> Animation | None:
    return FadeIn(mob, lag_ratio=ctx.mark_lag, **ctx.kwargs) if _has_points(mob) else None


def create(mob: Any, ctx: AnimationContext) -> Animation | None:
    return Create(mob, lag_ratio=ctx.mark_lag, **ctx.kwargs) if _has_points(mob) else None


def write(mob: Any, ctx: AnimationContext) -> Animation | None:
    return Write(mob, **ctx.kwargs) if _has_points(mob) else None


def grow(mob: Mobject, ctx: AnimationContext) -> Animation | None:
    marks: list[Animation] = [GrowFromCenter(m) for m in _marks(mob) if _has_points(m)]
    rest = _fade_others(mob)
    return _group(marks + rest, lag_ratio=ctx.mark_lag, run_time=1, **ctx.kwargs)


class GrowFromBaseline(Transform):
    """Introduce ``mobject`` by stretching it out of a line: along ``dim`` (1 =
    vertical, 0 = horizontal) from ``coordinate`` (scene units, clamped to the
    mobject's extent, so a bar grows from its end nearest the line)."""

    def __init__(self, mobject: Mobject, coordinate: float, dim: int = 1, **kwargs: Any):
        lo = mobject.get_critical_point(-_unit(dim))[dim]
        hi = mobject.get_critical_point(_unit(dim))[dim]
        self.dim = dim
        self.anchor = mobject.get_center().copy()
        self.anchor[dim] = float(np.clip(coordinate, min(lo, hi), max(lo, hi)))
        super().__init__(mobject, introducer=True, **kwargs)

    def create_target(self) -> Any:
        return self.mobject

    def create_starting_mobject(self) -> Any:
        start: Any = super().create_starting_mobject()
        start.stretch(0, self.dim, about_point=self.anchor)
        return start


def _unit(dim: int) -> np.ndarray:
    v = np.zeros(3)
    v[dim] = 1.0
    return v


def _zero_line(panel: Any, flipped: bool) -> float:
    """Scene coordinate (y; x under ``coord_flip``) of the value axis's zero."""
    params = panel.panel_params
    lo, hi = params.y_range
    zero = float(np.clip(0.0, min(lo, hi), max(lo, hi)))
    pt = panel.transform(params.x_range[0], zero)[0]
    return float(pt[0 if flipped else 1])


def grow_from_baseline(mob: Mobject, ctx: AnimationContext) -> Animation | None:
    dim = 0 if ctx.flipped else 1
    try:
        base = _zero_line(ctx.panel, ctx.flipped)
    except (AttributeError, TypeError):
        return grow_from_edge(mob, ctx)
    marks: list[Animation] = [GrowFromBaseline(m, base, dim) for m in _marks(mob) if _has_points(m)]
    rest = _fade_others(mob)
    return _group(marks + rest, lag_ratio=ctx.mark_lag, run_time=1, **ctx.kwargs)


def grow_from_edge(mob: Mobject, ctx: AnimationContext) -> Animation | None:
    dim = 0 if ctx.flipped else 1
    anims: list[Animation] = []
    for m in _marks(mob):
        if _has_points(m):
            anims.append(GrowFromBaseline(m, m.get_critical_point(-_unit(dim))[dim], dim))
    rest = _fade_others(mob)
    return _group(anims + rest, lag_ratio=ctx.mark_lag, run_time=1, **ctx.kwargs)


#: Animation factories by name, usable wherever a factory is accepted.
PRESETS: dict[str, Factory] = {
    "fade_in": fade_in,
    "create": create,
    "write": write,
    "grow": grow,
    "grow_from_baseline": grow_from_baseline,
    "grow_from_edge": grow_from_edge,
}


def _factory(f: str | Factory) -> Factory:
    if isinstance(f, str):
        try:
            return PRESETS[f]
        except KeyError:
            raise ValueError(f"unknown animation {f!r}; presets: {sorted(PRESETS)}") from None
    if not callable(f):
        raise TypeError(f"animation must be a preset name or a callable, got {f!r}")
    return f


def _reverse(factory: Factory) -> Factory:
    """``factory`` played backwards, removing the mobject at the end."""

    def hide(mob: Mobject, ctx: AnimationContext) -> Animation | None:
        rate = ctx.kwargs.pop("rate_func", smooth)
        anim = factory(mob, ctx)
        if anim is None:
            return None
        return AnimationGroup(
            anim, rate_func=lambda t: rate(1 - t), remover=True, run_time=anim.run_time
        )

    return hide


# -- registry -----------------------------------------------------------------------

#: ``{Geom class name: AnimationEntry}``; see :func:`register_animation`.
ANIMATIONS: dict[str, AnimationEntry] = {}


def register_animation(
    geom: type | str,
    reveal: str | Factory,
    hide: str | Factory | None = None,
    *,
    batched: Callable[[Mobject], bool] | None = None,
) -> AnimationEntry:
    """Set the animation for layers drawn by ``geom`` (a ``Geom`` class or class
    name; subclasses inherit it). ``reveal``/``hide`` are preset names
    (:data:`PRESETS`) or factories ``(mob, ctx) -> Animation``; ``hide=None``
    plays ``reveal`` backwards. Returns the entry."""
    name = geom if isinstance(geom, str) else geom.__name__
    rev = _factory(reveal)
    entry = AnimationEntry(rev, _factory(hide) if hide is not None else None, batched)
    ANIMATIONS[name] = entry
    return entry


def animation_for(geom: Any) -> AnimationEntry | None:
    """The registry entry for ``geom`` (instance, class or class name), walking
    the MRO; ``None`` when nothing in it is registered."""
    if geom is None:
        return None
    if isinstance(geom, str):
        return ANIMATIONS.get(geom)
    cls = geom if isinstance(geom, type) else type(geom)
    for klass in cls.__mro__:
        entry = ANIMATIONS.get(klass.__name__)
        if entry is not None:
            return entry
    return None


def _is_text(m: Mobject) -> bool:
    from manim import MarkupText, SingleStringMathTex, Text

    return isinstance(m, (Text, MarkupText, SingleStringMathTex))


def _infer(mob: Mobject) -> AnimationEntry:
    """Entry for a layer whose geom is unknown, from its marks."""
    marks = [m for m in _marks(mob) if _has_points(m)]
    if marks and all(_is_text(m) for m in marks):
        return AnimationEntry(write)
    leaves = [f for m in marks for f in m.family_members_with_points()]
    if leaves and all(
        isinstance(f, VMobject) and not _is_text(f) and f.get_fill_opacity() == 0 for f in leaves
    ):
        return AnimationEntry(create)
    return AnimationEntry(fade_in)


def _batched_raster(mob: Mobject) -> bool:
    return any(len(m.get_subpaths()) > 1 for m in _marks(mob) if isinstance(m, VMobject))


for _names, _preset in (
    (("GeomPoint", "GeomCount", "GeomJitter", "GeomDotplot"), "fade_in"),
    (
        (
            "GeomPath",  # GeomLine, GeomStep, GeomFunction, GeomSmooth, GeomQqLine
            "GeomAbline",
            "GeomVline",
            "GeomHline",
            "GeomSegment",
            "GeomCurve",
            "GeomSpoke",
            "GeomRug",
            "GeomLinerange",
            "GeomPointrange",
            "GeomErrorbar",
            "GeomErrorbarh",
            "GeomContour",
            "GeomDensity2d",
            "GeomQuantile",
        ),
        "create",
    ),
    (("GeomCol", "GeomBar", "GeomHistogram"), "grow_from_baseline"),
    (("GeomBoxplot",), "grow_from_edge"),
    (
        (
            "GeomRibbon",  # GeomArea
            "GeomDensity",
            "GeomViolin",
            "GeomRect",
            "GeomTile",
            "GeomPolygon",
            "GeomCrossbar",
            "GeomHex",
            "GeomBin2d",
        ),
        "fade_in",
    ),
    (("GeomText", "GeomLabel"), "write"),
):
    for _name in _names:
        register_animation(_name, _preset)
register_animation("GeomRaster", "fade_in", batched=_batched_raster)


# -- plot structure -----------------------------------------------------------------


def attach_geoms(plot: Any, spec: Any) -> Any:
    """Record ``{layer key: Geom}`` from the ``ggplot`` ``spec`` that ``plot``
    was built from as ``plot.layer_geoms`` (keys as in ``plot.layers``: the
    layer name, else its index). Returns ``plot``."""
    plot.layer_geoms = {
        (layer.name if layer.name is not None else i): layer.geom
        for i, layer in enumerate(spec.layers)
    }
    return plot


def _flipped(plot: Any) -> bool:
    return bool(getattr(plot.panel, "flipped", False))


def _layer_key(plot: Any, layer: Any) -> Any:
    """Resolve ``layer`` (name, key or position) to its key in ``plot.layers``."""
    if layer in dict.keys(plot.layers):
        return layer
    if isinstance(layer, int) and not isinstance(layer, bool):
        keys = list(plot.layers)
        if -len(keys) <= layer < len(keys):
            return keys[layer]
    raise KeyError(f"no layer {layer!r}; layers: {list(plot.layers)}")


def _geom(plot: Any, key: Any) -> Any:
    return (getattr(plot, "layer_geoms", None) or {}).get(key)


def _entry(plot: Any, key: Any, mob: Mobject) -> AnimationEntry:
    entry = animation_for(_geom(plot, key))
    return entry if entry is not None else _infer(mob)


def _parts(plot: Any, mob: Mobject) -> list[tuple[Mobject, Any]]:
    """``[(part, panel)]``: one per panel for a faceted layer."""
    parts = getattr(mob, "parts", None)
    if parts is None:
        return [(mob, plot.panel)]
    panels = plot.panels
    return [(part, panels.get(cell, plot.panel)) for cell, part in parts.items()]


def _layer_animation(
    plot: Any,
    layer: Any,
    *,
    hide: bool = False,
    animation: str | Factory | None = None,
    mark_lag: float = 0.05,
    panel_lag: float = 0.0,
    **kwargs: Any,
) -> Animation | None:
    key = _layer_key(plot, layer)
    mob = plot.layers[key]
    if animation is not None:
        factory = _factory(animation)
        entry = AnimationEntry(factory)
    else:
        entry = _entry(plot, key, mob)
    if hide:
        factory = entry.hide if entry.hide is not None else _reverse(entry.reveal)
    else:
        factory = entry.reveal
    run_time = kwargs.pop("run_time", None)
    anims: list[Animation] = []
    for part, panel in _parts(plot, mob):
        ctx = AnimationContext(
            plot, key, _geom(plot, key), panel, _flipped(plot), mark_lag, dict(kwargs)
        )
        anim = factory(part, ctx)
        if anim is not None:
            anims.append(anim)
    if not anims:
        return None
    if len(anims) == 1:
        out = anims[0]
        if run_time is not None:
            out.run_time = run_time
        return out
    group_kw: dict[str, Any] = {"lag_ratio": panel_lag}
    if run_time is not None:
        group_kw["run_time"] = run_time
    if hide:
        group_kw["remover"] = True
    return AnimationGroup(*anims, **group_kw)


def reveal_layer(plot: Any, layer: Any, **kwargs: Any) -> Animation:
    """Animation that draws one layer with its geom's registered animation.

    ``animation=`` overrides it (a preset name or factory); ``mark_lag``
    staggers marks, ``panel_lag`` panels; other keyword arguments
    (``run_time``, ``rate_func`` ...) go to the animation.
    """
    anim = _layer_animation(plot, layer, **kwargs)
    if anim is None:
        raise ValueError(f"layer {layer!r} has nothing to draw")
    return anim


def hide_layer(plot: Any, layer: Any, **kwargs: Any) -> Animation:
    """Animation that removes one layer (the registered ``hide``, by default the
    reveal played backwards). Keyword arguments as for :func:`reveal_layer`."""
    anim = _layer_animation(plot, layer, hide=True, **kwargs)
    if anim is None:
        raise ValueError(f"layer {layer!r} has nothing to draw")
    return anim


def mark(plot: Any, layer: Any, key: Any) -> Mobject:
    """The mobject drawn for the data row (or group) whose ``key`` aesthetic is
    ``key`` (the row index / group id when ``key`` is not mapped). Faceted
    layers search every panel. Raises ``KeyError`` listing the available keys,
    ``TypeError`` for a layer whose marks are batched."""
    k = _layer_key(plot, layer)
    mob = plot.layers[k]
    marks, keys = getattr(mob, "marks", None), getattr(mob, "keys", None)
    entry = animation_for(_geom(plot, k))
    batched = getattr(mob, "batched", False) or any(
        getattr(part, "batched", False)
        or (entry is not None and entry.batched is not None and entry.batched(part))
        for part, _ in _parts(plot, mob)
    )
    if marks is None or keys is None or batched:
        raise TypeError(
            f"layer {k!r} draws its marks in batches, not one mobject per data row; "
            "per-mark access is not available"
        )
    for m, mk in zip(marks, keys):
        if _same(mk, key):
            return m
    shown = ", ".join(repr(x) for x in keys[:20]) + (", ..." if len(keys) > 20 else "")
    raise KeyError(f"no mark with key {key!r} in layer {k!r}; keys: [{shown}]")


def _same(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


# -- the whole plot -----------------------------------------------------------------


def _panel_pieces(panel: Any) -> tuple[list, list, list]:
    """``(panel parts, axis lines/ticks, labels)`` of one CartesianPanel."""
    lines = [panel.axes, panel.x_ticks, panel.y_ticks]
    labels = [panel.x_tick_labels, panel.y_tick_labels, panel.x_title, panel.y_title]
    taken = {id(m) for m in lines + labels if m is not None}
    rest = [m for m in panel.submobjects if id(m) not in taken]
    return rest, lines, [m for m in labels if m is not None]


def _panel_stage(plot: Any, panel_lag: float) -> Animation | None:
    per_panel = []
    for panel in plot.panels.values():
        rest = _panel_pieces(panel)[0]
        bg: Any = getattr(panel, "background", None)
        anims: list[Animation] = [
            (FadeIn(m) if m is bg else Create(m, lag_ratio=0.02)) for m in rest if _has_points(m)
        ]
        per_panel.append(_group(anims))
    strips = plot.strips
    extra = [FadeIn(strips, lag_ratio=0.05)] if _has_points(strips) else []
    plot_bg: Any = getattr(plot, "background", None)
    first: list[Animation] = [FadeIn(plot_bg)] if _has_points(plot_bg) else []
    panels = _group(per_panel, lag_ratio=panel_lag)
    return _group([*first, panels, *extra])


def _axes_stage(plot: Any, panel_lag: float) -> Animation | None:
    per_panel = []
    for panel in plot.panels.values():
        _, lines, labels = _panel_pieces(panel)
        anims: list[Animation] = [Create(m) for m in lines if _has_points(m)]
        anims += [FadeIn(m, lag_ratio=0.05) for m in labels if _has_points(m)]
        per_panel.append(_group(anims))
    grid = plot.panel_grid
    titles = []
    if grid is not None:
        titles = [Write(m) for m in (grid.x_title, grid.y_title) if _has_points(m)]
    return _group([_group(per_panel, lag_ratio=panel_lag), *titles])


def _top_level(plot: Any, order: Sequence[str]) -> list[Mobject]:
    """The GGPlot submobjects covered by the stages in ``order``."""
    out: list[Mobject] = []
    if "panel" in order and "axes" in order:
        if plot.background is not None:
            out.append(plot.background)
        out.append(plot.panel_grid if plot.panel_grid is not None else plot.panel)
    if "layers" in order:
        out.extend(plot.layers.values())
    if "legend" in order:
        out.append(plot.legend)
    if "titles" in order:
        out.extend(m for m in (plot.title, plot.subtitle, plot.caption) if m is not None)
    return out


class _PlotCreation(AnimationGroup):
    """The :func:`create_plot` group. When every part of the plot was animated,
    the scene ends up holding the ``GGPlot`` itself (not its parts), so later
    ``plot.animate`` calls and ``scene.remove(plot)`` behave as expected."""

    def __init__(self, *anims: Animation, plot: Any, parts: list[Mobject], **kwargs: Any):
        self.plot = plot
        self._parts = parts
        super().__init__(*anims, **kwargs)

    def clean_up_from_scene(self, scene: Any) -> None:
        super().clean_up_from_scene(scene)
        if self._parts:
            covered = {id(m) for m in self._parts}
            if all(id(m) in covered for m in self.plot.submobjects):
                family = {id(m) for m in self.plot.get_family()} | {id(self.mobject)}
                scene.mobjects = [m for m in scene.mobjects if id(m) not in family]
                scene.add(self.plot)


def create_plot(
    plot: Any,
    *,
    lag_ratio: float = 1.0,
    run_time: float | None = None,
    order: Sequence[str] = DEFAULT_ORDER,
    panel_lag: float = 0.0,
    mark_lag: float = 0.05,
    animations: Mapping[Any, str | Factory] | None = None,
) -> Animation:
    """One animation that reveals a built plot piece by piece.

    Parameters
    ----------
    plot : :class:`~ggmanim.mobjects.GGPlot`, or a ``ggplot`` spec (built here,
        with :func:`attach_geoms`; the plot is then ``animation.plot``)
    lag_ratio : float
        Overlap of consecutive stages: ``1`` plays them one after another, ``0``
        all at once.
    run_time : float, optional
        Total run time (default: one second per stage, less the overlap).
    order : sequence of str
        Stages, from ``"panel"`` (background, grid, strips), ``"axes"`` (axis
        lines, ticks, tick labels, axis titles), ``"layers"`` (one stage per
        layer, in spec order, each with its geom's animation), ``"legend"``
        and ``"titles"``. Parts whose stage is left out are not shown.
    panel_lag : float
        Stagger of the panels of a faceted plot (``0``: together).
    mark_lag : float
        ``lag_ratio`` between the marks of a layer.
    animations : mapping, optional
        ``{layer: preset name or factory}`` overriding the registry.
    """
    if hasattr(plot, "build") and not isinstance(plot, Mobject):
        plot = attach_geoms(plot.build(), plot)
    unknown = [s for s in order if s not in _STAGES]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; use some of {list(DEFAULT_ORDER)}")
    overrides = dict(animations or {})
    stages: list[Animation | None] = []
    for stage in order:
        if stage == "panel":
            stages.append(_panel_stage(plot, panel_lag))
        elif stage == "axes":
            stages.append(_axes_stage(plot, panel_lag))
        elif stage == "layers":
            for key in plot.layers:
                stages.append(
                    _layer_animation(
                        plot,
                        key,
                        animation=overrides.get(key),
                        mark_lag=mark_lag,
                        panel_lag=panel_lag,
                    )
                )
        elif stage == "legend" and _has_points(plot.legend):
            stages.append(FadeIn(plot.legend, lag_ratio=0.05))
        elif stage == "titles":
            titles = [plot.title, plot.subtitle, plot.caption]
            stages.append(_group([Write(m) for m in titles if _has_points(m)]))
    anims = [a for a in stages if a is not None]
    if not anims:
        raise ValueError("nothing to animate")
    kwargs: dict[str, Any] = {"lag_ratio": lag_ratio}
    if run_time is not None:
        kwargs["run_time"] = run_time
    return _PlotCreation(*anims, plot=plot, parts=_top_level(plot, order), **kwargs)
