"""geom_density: smoothed density curves (stat_density) drawn as areas.

A `GeomArea` fed by
`StatDensity` with ``y = after_stat("density")``
mapped by default (``after_stat("count")``, ``"scaled"`` ... also work). As in
ggplot2 the outline (``outline_type="upper"``) is drawn in ``color`` (the theme
foreground by default) and the area is unfilled unless ``fill`` is mapped or
set; ``alpha`` applies to the fill only, so ``geom_density(aes(fill="g"),
alpha=0.4)`` gives overlapping translucent densities. The default position is
``"identity"`` (``stat_density()`` stacks).

``orientation="y"`` estimates the density of ``y`` and draws it along the y
axis (``x = after_stat("density")``). Without ``orientation``, a density with
``y`` but no ``x`` mapped -- in the layer *or the plot* mapping -- is horizontal
too: the default mapping is added by [`DensityLayer`][ggmanim.geoms.DensityLayer] when the layer is
merged with the plot mapping.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import pandas as pd
from manim import VMobject

from ..aes import Aes
from ..layer import Layer
from ..registry import register
from ..stats.boxplot import resolve_flipped
from ..stats.density import StatDensity, density_mapping
from ..theme import Theme, theme_default
from ._rect import filled_key
from .area import GeomArea, _check_outline_type
from .base import make_layer

__all__ = ["DensityLayer", "GeomDensity", "geom_density"]


def _is_flipped(data: pd.DataFrame, params: Mapping[str, Any]) -> bool:
    return resolve_flipped(data, None, params.get("orientation"), main_is_orthogonal=False)


class DensityLayer(Layer):
    """A [`Layer`][ggmanim.layer.Layer] that adds the density default mapping
    (`density_mapping()`) *after* merging with the
    plot mapping, so the orientation follows the plot-level ``aes`` too:
    ``ggplot(df, aes(y="v")) + geom_density()`` is horizontal."""

    @classmethod
    def from_layer(cls, layer: Layer) -> DensityLayer:
        """``layer`` as a [`DensityLayer`][ggmanim.geoms.DensityLayer] (same fields)."""
        return cls(**{f.name: getattr(layer, f.name) for f in dataclasses.fields(layer)})

    def resolve_mapping(self, plot_mapping: Aes | None) -> Aes:
        return density_mapping(super().resolve_mapping(plot_mapping), self.params)


class GeomDensity(GeomArea):
    """Density areas: one mark per group, like [`GeomArea`][ggmanim.geoms.GeomArea]."""

    default_stat = "density"
    default_position = "identity"
    default_aes = {
        "color": None,
        "fill": None,
        "linewidth": 3.0,
        "alpha": 1.0,
        "linetype": "solid",
    }
    extra_params: tuple[str, ...] = ("outline_type", "orientation")

    def is_flipped(self, data: pd.DataFrame, params: Any = None) -> bool:
        """Densities: ``y`` without ``x`` is horizontal too (flipping is done by
        [`GeomArea`][ggmanim.geoms.GeomArea]'s ``setup_data``/``draw_panel``)."""
        return _is_flipped(data, params or {})

    def draw_panel(
        self, data, panel_params, coord, theme: Theme | None = None, outline_type=None, **params
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        data = self.use_defaults(data, params)
        fg = theme.get_foreground()
        unfilled = data["fill"].map(lambda v: v is None or (isinstance(v, float) and v != v))
        data = data.assign(
            alpha=data["alpha"].where(~unfilled, 0.0),
            color=data["color"].map(lambda v: fg if v is None else v),
        )
        params = {k: v for k, v in params.items() if k not in ("alpha", "fill", "color")}
        return super().draw_panel(
            data, panel_params, coord, theme, outline_type=outline_type, **params
        )

    def draw_key(self, data_row, theme: Theme | None = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        row = self.key_aes(data_row)
        if row.get("color") is None:
            row["color"] = theme.get_foreground()
        if row.get("fill") is None:
            row["alpha"] = 0.0
        return filled_key(row, theme, default_linewidth=2.0)


@register
def geom_density(
    mapping=None,
    data=None,
    *,
    outline_type: str = "upper",
    stat="density",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Kernel density curves (ggplot2's ``geom_density``); see
    `ggmanim.geoms.density` and `ggmanim.stats.density`.

    Stat params: ``bw`` (``"nrd0"``, ``"nrd"``/``"scott"``, ``"ucv"``, ``"bcv"``,
    ``"SJ"``/``"sj-dpi"``, a number or a callable), ``adjust``, ``kernel``, ``n`` (512), ``trim``, ``bounds``.
    ``outline_type`` as for `geom_ribbon()`."""
    params["outline_type"] = _check_outline_type(outline_type)
    layer = make_layer(
        GeomDensity(),
        mapping,
        data,
        stat=stat,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
    return DensityLayer.from_layer(layer) if isinstance(layer.stat, StatDensity) else layer
