"""The ``ggplot`` plot specification.

A ``ggplot`` is an immutable description of a plot: data, default mapping,
layers, scales, coord, facet, theme and labels. Components are combined with
:meth:`ggplot.add`, which returns a *new* plot (the DataFrame is shared by
reference). Every component implements ``add_to(plot) -> plot`` and uses the
``_with_*`` helpers below.

Chaining sugar is generated from :data:`ggmanim.registry.COMPONENTS` at call
time: ``p.geom_point(alpha=.5)`` is ``p.add(geom_point(alpha=.5))``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import pandas as pd

from .aes import Aes, standardise_aes_name
from .layer import Layer
from .registry import COMPONENTS, register

__all__ = ["ggplot", "GGManim", "Labels", "labs", "xlab", "ylab"]

_FIELDS = ("data", "mapping", "layers", "scales", "coord", "facet", "theme", "labels", "guide_specs")


class ggplot:
    """Immutable plot spec. Build with ``.add(...)`` or generated chaining methods."""

    __slots__ = _FIELDS

    data: Optional[pd.DataFrame]
    mapping: Aes
    layers: Tuple[Layer, ...]
    scales: Tuple[Any, ...]
    coord: Any
    facet: Any
    theme: Any
    labels: Mapping[str, Any]
    guide_specs: Mapping[str, Any]

    def __init__(self, data: Optional[pd.DataFrame] = None, mapping: Optional[Mapping] = None):
        if isinstance(data, Mapping) and not isinstance(data, pd.DataFrame) and mapping is None:
            # ggplot(mapping=...) written positionally as ggplot(aes(...))
            data, mapping = None, data
        _set = object.__setattr__
        _set(self, "data", data)
        _set(self, "mapping", mapping if isinstance(mapping, Aes) else Aes(mapping))
        _set(self, "layers", ())
        _set(self, "scales", ())
        _set(self, "coord", None)
        _set(self, "facet", None)
        _set(self, "theme", None)
        _set(self, "labels", _FrozenDict())
        _set(self, "guide_specs", _FrozenDict())

    # -- immutability ------------------------------------------------------
    def __setattr__(self, name, value):
        raise AttributeError(
            f"ggplot is immutable; use .add(...) or a component to get a new plot"
        )

    def __delattr__(self, name):
        raise AttributeError("ggplot is immutable")

    def _replace(self, **changes) -> "ggplot":
        new = object.__new__(type(self))
        for f in _FIELDS:
            object.__setattr__(new, f, changes.pop(f) if f in changes else getattr(self, f))
        if changes:
            raise TypeError(f"unknown ggplot fields: {sorted(changes)}")
        return new

    def __copy__(self):
        return self._replace()

    def __reduce__(self):
        return (_rebuild, (type(self), {f: getattr(self, f) for f in _FIELDS}))

    # -- composition -------------------------------------------------------
    def add(self, *components, name: Optional[str] = None) -> "ggplot":
        """Return a new plot with ``components`` added in order.

        ``name`` names the (single) layer added by these components.
        """
        plot = self
        n_before = len(self.layers)
        for c in components:
            if c is None:
                continue
            add_to = getattr(c, "add_to", None)
            if add_to is None:
                raise TypeError(
                    f"cannot add {c!r} to a ggplot: it has no add_to(plot) method"
                )
            new = add_to(plot)
            if not isinstance(new, ggplot):
                raise TypeError(
                    f"{type(c).__name__}.add_to must return a ggplot, got {type(new).__name__}"
                )
            plot = new
        if name is not None:
            added = plot.layers[n_before:]
            if len(added) != 1:
                raise ValueError(
                    f"name={name!r} requires exactly one layer to be added, got {len(added)}"
                )
            if any(l.name == name for l in plot.layers[:n_before]):
                raise ValueError(f"a layer named {name!r} already exists")
            plot = plot._replace(layers=plot.layers[:n_before] + (added[0].replace(name=name),))
        return plot

    # -- helpers used by components ---------------------------------------
    def _with_layer(self, layer: Layer) -> "ggplot":
        if layer.name is not None and any(l.name == layer.name for l in self.layers):
            raise ValueError(f"a layer named {layer.name!r} already exists")
        return self._replace(layers=self.layers + (layer,))

    def _with_scale(self, scale) -> "ggplot":
        """Add a scale; any earlier scale sharing an aesthetic is replaced."""
        new_aes = set(_scale_aesthetics(scale))
        kept = tuple(s for s in self.scales if not (set(_scale_aesthetics(s)) & new_aes))
        return self._replace(scales=kept + (scale,))

    def _with_coord(self, coord) -> "ggplot":
        return self._replace(coord=coord)

    def _with_facet(self, facet) -> "ggplot":
        return self._replace(facet=facet)

    def _with_theme(self, theme) -> "ggplot":
        """Replace the theme if ``theme`` is complete or none is set, else merge.

        Merging uses ``old.merge(new)`` when the current theme provides it.
        """
        old = self.theme
        if old is None or getattr(theme, "complete", False) or not hasattr(old, "merge"):
            return self._replace(theme=theme)
        return self._replace(theme=old.merge(theme))

    def _with_labels(self, **labels) -> "ggplot":
        new = dict(self.labels)
        for k, v in labels.items():
            new[standardise_aes_name(k)] = v
        return self._replace(labels=_FrozenDict(new))

    def _with_guides(self, mapping: Mapping[str, Any]) -> "ggplot":
        """Merge ``mapping`` (aesthetic -> guide spec, e.g. ``"none"``) into the guides."""
        new = dict(self.guide_specs)
        for k, v in mapping.items():
            new[standardise_aes_name(k)] = v
        return self._replace(guide_specs=_FrozenDict(new))

    @property
    def guides(self) -> "_GuidesView":
        """Read-only mapping aesthetic -> guide (``"legend"``/``"none"``) set by the
        ``guides()`` component. Also callable as chaining sugar:
        ``p.guides(color="none") == p.add(guides(color="none"))``."""
        return _GuidesView(self)

    # -- queries -----------------------------------------------------------
    def get_scale(self, aesthetic: str):
        """The user-specified scale for ``aesthetic`` or ``None``."""
        aesthetic = standardise_aes_name(aesthetic)
        for s in self.scales:
            if aesthetic in _scale_aesthetics(s):
                return s
        return None

    def layer(self, name: str) -> Layer:
        for l in self.layers:
            if l.name == name:
                return l
        raise KeyError(name)

    # -- generated chaining -------------------------------------------------
    def __getattr__(self, attr: str):
        # Only called when normal lookup fails. Never treat private names as
        # components (copy/pickle/IPython probe dunders).
        if attr.startswith("_"):
            raise AttributeError(attr)
        component = COMPONENTS.get(attr)
        if component is None:
            raise AttributeError(
                f"'ggplot' object has no attribute {attr!r} (no registered component {attr!r})"
            )
        plot = self

        def method(*args, **kwargs):
            name = kwargs.pop("name", None)
            layer_id = kwargs.pop("layer_id", None)  # legacy GGManim spelling
            if name is not None and layer_id is not None:
                raise TypeError("pass only one of name= and layer_id=")
            return plot.add(component(*args, **kwargs), name=name if name is not None else layer_id)

        method.__name__ = attr
        method.__qualname__ = f"ggplot.{attr}"
        method.__doc__ = getattr(component, "__doc__", None)
        return method

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(COMPONENTS))

    def __repr__(self):
        data = "None" if self.data is None else f"DataFrame{getattr(self.data, 'shape', '')}"
        parts = [f"data={data}", f"mapping={self.mapping!r}"]
        if self.layers:
            parts.append("layers=[" + ", ".join(
                (l.name or type(l.geom).__name__) for l in self.layers) + "]")
        if self.scales:
            parts.append("scales=[" + ", ".join(type(s).__name__ for s in self.scales) + "]")
        for f in ("coord", "facet", "theme"):
            v = getattr(self, f)
            if v is not None:
                parts.append(f"{f}={type(v).__name__}")
        if self.labels:
            parts.append(f"labels={dict(self.labels)!r}")
        return f"ggplot({', '.join(parts)})"


#: Legacy name.
GGManim = ggplot


def _rebuild(cls, state):
    new = object.__new__(cls)
    for k, v in state.items():
        object.__setattr__(new, k, v)
    return new


def _scale_aesthetics(scale):
    aes = getattr(scale, "aesthetics", ())
    if isinstance(aes, str):
        aes = (aes,)
    return tuple(standardise_aes_name(a) for a in aes)


class _FrozenDict(dict):
    """A dict that refuses mutation (labels on an immutable plot)."""

    def _immutable(self, *a, **k):
        raise TypeError("plot labels are immutable; use .labs(...)")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable  # type: ignore
    __ior__ = _immutable  # type: ignore

    def __reduce__(self):
        return (_FrozenDict, (dict(self),))


class _GuidesView(Mapping):
    """``plot.guide_specs`` as a mapping that is also the ``guides`` chaining method."""

    __slots__ = ("_plot",)

    def __init__(self, plot: ggplot):
        self._plot = plot

    def __getitem__(self, key):
        return self._plot.guide_specs[key]

    def __iter__(self):
        return iter(self._plot.guide_specs)

    def __len__(self):
        return len(self._plot.guide_specs)

    def __call__(self, *args, **kwargs) -> ggplot:
        component = COMPONENTS["guides"]
        return self._plot.add(component(*args, **kwargs))

    def __repr__(self):
        return f"guides({dict(self._plot.guide_specs)!r})"


# -- label components --------------------------------------------------------


class Labels:
    """Component setting axis/legend titles and plot title/subtitle/caption."""

    __slots__ = ("labels",)

    def __init__(self, **labels):
        self.labels: Dict[str, Any] = {standardise_aes_name(k): v for k, v in labels.items()}

    def add_to(self, plot: ggplot) -> ggplot:
        return plot._with_labels(**self.labels)

    def __repr__(self):
        return f"labs({', '.join(f'{k}={v!r}' for k, v in self.labels.items())})"


@register
def labs(**labels) -> Labels:
    """Set labels: ``labs(x=..., y=..., color=..., title=..., subtitle=..., caption=...)``.

    Values may be strings or ready-made manim Mobjects; ``None`` suppresses a label.
    """
    return Labels(**labels)


@register
def xlab(label) -> Labels:
    """Set the x-axis title."""
    return Labels(x=label)


@register
def ylab(label) -> Labels:
    """Set the y-axis title."""
    return Labels(y=label)
