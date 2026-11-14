"""Live plots: data updates in place and ``ValueTracker``-driven layers.

::

    from ggmanim.live import LivePlot, always_rebuild, build_live

    plot = build_live(spec)                  # spec has fixed limits (xlim/ylim)
    self.add(plot)
    plot.add_updater(lambda m: m.update_data(next_frame()))

    a = ValueTracker(1)
    spec = (ggplot().geom_function(fun=lambda x, a=a: a.get_value() * np.sin(x))
            + xlim(0, 6.3) + ylim(-2, 2))
    self.add(always_rebuild(spec))           # redraws the curve every frame
    self.play(a.animate.set_value(2))

The ``ggplot`` spec stays immutable: :meth:`ggplot.with_data` gives a copy with
new plot data. The built :class:`~ggmanim.mobjects.GGPlot` is what changes.

Cost model
----------
* **Full rebuild** (``p.with_data(df).build()``, or manim's ``always_redraw``
  around it): every step of :mod:`ggmanim.build` runs, including the axes, tick
  labels, axis titles and legends, i.e. all the text. Text is by far the most
  expensive part of a build; do this between ``play`` calls, or use
  :func:`ggmanim.transition.transition` to morph between two builds.
* **In place** (:func:`update_data`, :meth:`LivePlot.refresh`,
  :func:`always_rebuild`): the data steps of the pipeline run again (aesthetics,
  stats, positions, scale mapping: numpy only), then only the updated layers'
  geoms draw against the *existing* panel. Nothing textual is rebuilt: the
  panel, axes, legend and titles are untouched. Each existing mark is then
  updated from the freshly drawn one, keyed by ``mob.keys`` (the ``key``
  aesthetic, else the row index / group id): point and text marks are moved
  (``move_to`` the new centre, so their size survives a rescaled plot) and
  take the new colours; other marks (lines, paths, bars, polygons) take the new
  outline with ``Mobject.become``. The mark objects keep their identity, so
  updaters, ``ValueTracker`` bindings and references from ``plot.mark(...)``
  remain valid.
* When the set of keys changes (rows added or removed, a mark clipped away) or
  a layer's marks are batched (``fast=True``), that layer's mobject is
  *replaced* by the newly drawn one (``plot.layers[key]`` and the GGPlot's
  submobject are swapped; the old mark objects are dropped). Still no text.

The panel is frozen at the first build, so the position scales must not
depend on the data: :func:`update_data` and :func:`always_rebuild` require
**fixed limits** on x and y, i.e. explicit scale limits (``xlim()``,
``scale_x_continuous(limits=...)``, discrete ``limits=``) or
``coord_cartesian(xlim=..., ylim=...)`` with both ends given; they raise
:class:`~ggmanim.build.BuildError` otherwise. :func:`update_data` also requires
the identity stat on the layers it updates (a stat such as ``stat_bin`` would
change the rows, not move them); :func:`always_rebuild` redraws any stat
(``geom_function``, smooths ...). Non-position scales (colour, size) are
retrained on every update but the legend is not redrawn; fix their
``limits`` too when the legend must stay true. Faceted plots are not
supported.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd
from manim import Mobject

from .build import (
    BuildError,
    LayerData,
    apply_positions,
    assign_panels,
    compute_stats,
    draw_layers,
    map_non_position,
    resolve_layers,
    setup_geom_data,
    setup_position_scales,
    train_position_scales,
)
from .scales.discrete_position import ScaleDiscretePosition
from .stats import get_stat
from .stats.identity import StatIdentity
from .theme import theme_default

__all__ = [
    "LivePlot",
    "always_rebuild",
    "build_live",
    "check_fixed_limits",
    "refresh",
    "update_data",
]

#: Geom classes (by name, matched along the MRO) whose marks are glyphs placed at
#: a point: updated with ``move_to`` so their size is kept.
_CENTRED_GEOMS = frozenset({"GeomPoint", "GeomText", "GeomLabel"})


# -- checks ------------------------------------------------------------------------


def _fixed(lim: Any) -> bool:
    if lim is None or callable(lim):
        return False
    try:
        vals = list(lim)
    except TypeError:
        return False
    return len(vals) >= 1 and all(v is not None for v in vals)


def check_fixed_limits(spec: Any) -> None:
    """Raise :class:`~ggmanim.build.BuildError` unless both position axes of
    ``spec`` have limits that do not depend on the data (explicit scale limits,
    or ``coord_cartesian`` limits with both ends)."""
    if getattr(spec, "facet", None) is not None and not getattr(spec.facet, "is_null", True):
        raise BuildError("live updates do not support faceted plots")
    coord = getattr(spec, "coord", None)
    missing = []
    for axis in ("x", "y"):
        if _fixed(getattr(coord, f"{axis}lim", None)):
            continue
        scale = spec.get_scale(axis)
        lim: Any = getattr(scale, "limits", None)
        if scale is not None and _fixed(lim):
            if isinstance(scale, ScaleDiscretePosition) or len(list(lim)) == 2:
                continue
        missing.append(axis)
    if missing:
        names = " and ".join(missing)
        raise BuildError(
            f"live updates need fixed {names} limits (the panel is not rebuilt): add "
            f"{', '.join(m + 'lim(lo, hi)' for m in missing)}, scale limits=, or "
            "coord_cartesian(xlim=..., ylim=...) with both ends"
        )


def _check_identity(lds: Iterable[LayerData]) -> None:
    for ld in lds:
        stat = get_stat(ld.layer.stat) if ld.layer.stat is not None else None
        if stat is not None and not isinstance(stat, StatIdentity):
            raise BuildError(
                f"{ld.label}: update_data needs stat='identity' (got {type(stat).__name__}); "
                "use always_rebuild()/LivePlot.refresh() for computed layers"
            )


# -- building ----------------------------------------------------------------------


def build_live(spec: Any, *, theme: Any = None) -> Any:
    """``spec.build(theme=theme)`` with the spec and theme recorded on the
    :class:`~ggmanim.mobjects.GGPlot` (``plot.spec``, ``plot.live_theme``) so
    :meth:`~ggmanim.mobjects.GGPlot.update_data` can be called without a spec."""
    plot = spec.build(theme=theme)
    plot.spec = spec
    plot.live_theme = theme
    return plot


def _resolve_spec(plot: Any, spec: Any) -> Any:
    spec = spec if spec is not None else getattr(plot, "spec", None)
    if spec is None:
        raise BuildError(
            "this GGPlot does not know its spec: build it with ggmanim.live.build_live(spec) "
            "(or LivePlot(spec)), or pass spec="
        )
    return spec


def _layer_data(spec: Any) -> list[LayerData]:
    """Steps 1-6 of :func:`ggmanim.build.build_plot` (no panel, no text)."""
    lds = resolve_layers(spec)
    assign_panels(lds, spec)
    pos_scales = setup_position_scales(spec, lds)
    compute_stats(lds, pos_scales)
    map_non_position(spec, lds)
    setup_geom_data(lds, pos_scales)
    apply_positions(lds, pos_scales)
    train_position_scales(lds, pos_scales)
    return lds


def _select(plot: Any, lds: list[LayerData], layers: Any, default: Callable) -> list[LayerData]:
    if layers is None:
        return [ld for ld in lds if default(ld)]
    if isinstance(layers, (str, int)):
        layers = [layers]
    keys = list(plot.layers)
    out = []
    for want in layers:
        match = [ld for ld in lds if ld.key == want]
        if not match and isinstance(want, int) and not isinstance(want, bool):
            if -len(keys) <= want < len(keys):
                match = [ld for ld in lds if ld.key == keys[want]]
        if not match:
            raise KeyError(f"no layer {want!r}; layers: {keys}")
        out.extend(match)
    return out


# -- applying ------------------------------------------------------------------------


def _centred(geom: Any) -> bool:
    return any(c.__name__ in _CENTRED_GEOMS for c in type(geom).__mro__)


def _keyable(mob: Mobject) -> bool:
    return (
        getattr(mob, "marks", None) is not None
        and getattr(mob, "keys", None) is not None
        and not getattr(mob, "batched", False)
    )


def _same_keys(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False
    try:
        return all(bool(x == y) for x, y in zip(a, b))
    except (TypeError, ValueError):
        return False


def _update_mark(old: Mobject, new: Mobject, centred: bool) -> None:
    if centred:
        old.move_to(new.get_center())
        if len(old.get_family()) == len(new.get_family()):
            for a, b in zip(old.get_family(), new.get_family()):
                a.interpolate_color(a, b, 1)
        return
    old.become(new)


def _replace_layer(plot: Any, key: Any, new: Mobject) -> None:
    old = plot.layers[key]
    try:
        i = plot.submobjects.index(old)
    except ValueError:
        plot.add(new)
    else:
        plot.submobjects[i] = new
    plot.layers[key] = new


def _apply(plot: Any, lds: list[LayerData], spec: Any) -> list[Any]:
    """Draw ``lds`` on ``plot``'s panel and update ``plot`` in place.
    Returns the keys of the layers that had to be replaced."""
    if not lds:
        return []
    theme = getattr(plot, "live_theme", None)
    if theme is None:
        theme = spec.theme if spec.theme is not None else theme_default()
    replaced = []
    for (key, new), ld in zip(draw_layers(lds, plot.panel, theme), lds):
        old = plot.layers[key]
        old_keys: Any = getattr(old, "keys", None)
        new_keys: Any = getattr(new, "keys", None)
        if _keyable(old) and _keyable(new) and _same_keys(list(old_keys), list(new_keys)):
            centred = _centred(ld.layer.geom)
            for o, n in zip(getattr(old, "marks", ()), getattr(new, "marks", ())):
                _update_mark(o, n, centred)
        else:
            _replace_layer(plot, key, new)
            replaced.append(key)
        plot.layer_data[key] = ld.data
    return replaced


def update_data(plot: Any, data: pd.DataFrame, layers: Any = None, *, spec: Any = None) -> Any:
    """Move ``plot``'s marks to where the rows of ``data`` (the new plot data)
    put them, without rebuilding axes, text or legends.

    Parameters
    ----------
    plot : :class:`~ggmanim.mobjects.GGPlot` built with :func:`build_live` (or
        pass ``spec``).
    data : the new plot-level data (layers with their own data frame keep it).
    layers : layer key(s) to update; default every layer that draws the plot data
        (``data=None`` or a callable of it).
    spec : the ``ggplot`` the plot was built from; default ``plot.spec``.

    Returns ``plot``. Raises :class:`~ggmanim.build.BuildError` when the limits
    are not fixed or an updated layer has a non-identity stat. See the module
    docstring for what is updated in place and what is replaced.
    """
    base = _resolve_spec(plot, spec)
    check_fixed_limits(base)
    new_spec = base.with_data(data)
    lds = _layer_data(new_spec)
    chosen = _select(
        plot,
        lds,
        layers,
        lambda ld: ld.layer.data is None or not isinstance(ld.layer.data, pd.DataFrame),
    )
    _check_identity(chosen)
    _apply(plot, chosen, new_spec)
    plot.spec = new_spec
    return plot


def refresh(plot: Any, spec: Any = None, layers: Any = None) -> Any:
    """Re-run the data steps of ``spec`` (default ``plot.spec``) and redraw
    ``layers`` (default all) on the existing panel, in place where the keys
    match. Any stat is allowed; this is what makes a ``geom_function`` whose
    ``fun`` reads a ``ValueTracker`` follow the tracker. Returns ``plot``."""
    spec = _resolve_spec(plot, spec)
    check_fixed_limits(spec)
    lds = _layer_data(spec)
    _apply(plot, _select(plot, lds, layers, lambda ld: True), spec)
    plot.spec = spec
    return plot


# -- helpers -----------------------------------------------------------------------


class LivePlot:
    """A built plot plus the spec (or spec factory) it follows.

    ``spec`` is a ``ggplot`` or a zero-argument callable returning one (called on
    every :meth:`refresh`, e.g. to read ``ValueTracker`` values into the data).
    The plot is built once (:attr:`plot`); afterwards only layers are redrawn.

    ::

        live = LivePlot(lambda: ggplot(frame(t.get_value()), aes("x", "y"))
                        .geom_point() + xlim(0, 1) + ylim(0, 1))
        self.add(live.plot)
        live.plot.add_updater(live.updater())
        self.play(t.animate.set_value(10), run_time=4)
    """

    def __init__(self, spec: Any, *, theme: Any = None, layers: Any = None):
        self.spec = spec
        self.layers = layers
        first = self._spec()
        check_fixed_limits(first)
        self.plot = build_live(first, theme=theme)

    def _spec(self) -> Any:
        return self.spec() if callable(self.spec) else self.spec

    def refresh(self, layers: Any = None) -> Any:
        """Redraw ``layers`` (default the constructor's, else all) from the
        current spec. Returns :attr:`plot`."""
        return refresh(self.plot, self._spec(), layers if layers is not None else self.layers)

    def update_data(self, data: pd.DataFrame, layers: Any = None) -> Any:
        """:func:`update_data` on :attr:`plot`."""
        return update_data(self.plot, data, layers, spec=self.plot.spec)

    def updater(self, layers: Any = None) -> Callable[[Mobject], None]:
        """A ``mob.add_updater`` callback that calls :meth:`refresh`."""

        def _update(_mob: Mobject) -> None:
            self.refresh(layers)

        return _update


def always_rebuild(spec_fn: Any, layers: Any = None, *, theme: Any = None) -> Any:
    """Like manim's ``always_redraw`` for a plot, but the panel (axes, text,
    legend) is built once and cached; every frame only ``layers`` (default all)
    are recomputed and redrawn in place (:func:`refresh`).

    ``spec_fn`` is a ``ggplot`` (its layers may read ``ValueTracker`` s, e.g.
    ``geom_function(fun=lambda x, a=tracker: a.get_value() * x)``) or a callable
    returning one. Returns the :class:`~ggmanim.mobjects.GGPlot` with the updater
    attached; ``plot.live`` is the :class:`LivePlot`."""
    live = LivePlot(spec_fn, theme=theme, layers=layers)
    live.plot.live = live
    live.plot.add_updater(live.updater())
    return live.plot
