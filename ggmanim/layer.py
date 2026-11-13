"""Layer: geom + stat + position + data + mapping + params."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

import pandas as pd

from .aes import Aes

if TYPE_CHECKING:  # pragma: no cover
    from .plot import ggplot

__all__ = ["Layer"]

LayerData: TypeAlias = pd.DataFrame | Callable[[pd.DataFrame], pd.DataFrame] | None


@dataclass(frozen=True, eq=False)
class Layer:
    """One layer of a plot.

    ``data`` is ``None`` (inherit the plot data), a DataFrame, or a callable
    applied to the plot data. ``mapping`` is merged over the plot mapping when
    ``inherit_aes`` is true. ``name`` is the key of the layer in the built
    mobject (``plot.layers[name]``).
    """

    geom: Any = None
    stat: Any = None
    position: Any = None
    data: LayerData = None
    mapping: Aes = field(default_factory=Aes)
    params: dict[str, Any] = field(default_factory=dict)
    inherit_aes: bool = True
    name: str | None = None

    def __post_init__(self):
        if self.mapping is None:
            object.__setattr__(self, "mapping", Aes())
        elif not isinstance(self.mapping, Aes):
            object.__setattr__(self, "mapping", Aes(self.mapping))
        object.__setattr__(self, "params", dict(self.params or {}))
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError(f"layer name must be a str or None, got {self.name!r}")

    # -- component protocol ------------------------------------------------
    def add_to(self, plot: ggplot) -> ggplot:
        return plot._with_layer(self)

    # -- helpers -----------------------------------------------------------
    def replace(self, **changes) -> Layer:
        return dataclasses.replace(self, **changes)

    def resolve_data(self, plot_data: pd.DataFrame | None) -> pd.DataFrame | None:
        """The data this layer draws from (never uses DataFrame truthiness)."""
        if self.data is None:
            return plot_data
        if callable(self.data) and not isinstance(self.data, pd.DataFrame):
            return self.data(plot_data)
        return self.data

    def resolve_mapping(self, plot_mapping: Aes | None) -> Aes:
        """Plot mapping overridden by the layer mapping (if ``inherit_aes``)."""
        if self.inherit_aes and plot_mapping is not None:
            base = plot_mapping if isinstance(plot_mapping, Aes) else Aes(plot_mapping)
            return base.merge(self.mapping).defined()
        return self.mapping.defined()

    def __repr__(self):
        def short(o):
            return type(o).__name__ if o is not None and not isinstance(o, str) else repr(o)

        data = (
            "None"
            if self.data is None
            else (
                f"DataFrame{self.data.shape}" if isinstance(self.data, pd.DataFrame) else "callable"
            )
        )
        return (
            f"Layer(name={self.name!r}, geom={short(self.geom)}, stat={short(self.stat)}, "
            f"position={short(self.position)}, data={data}, mapping={self.mapping!r}, "
            f"params={self.params!r}, inherit_aes={self.inherit_aes})"
        )
