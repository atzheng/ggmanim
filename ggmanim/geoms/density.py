"""geom_density: smoothed density curves (stat_density) drawn as areas.

A :class:`~ggmanim.geoms.area.GeomArea` fed by
:class:`~ggmanim.stats.density.StatDensity` with ``y = after_stat("density")``
mapped by default (``after_stat("count")``, ``"scaled"`` ... also work). As in
ggplot2 the outline (``outline_type="upper"``) is drawn in ``color`` (the theme
foreground by default) and the area is unfilled unless ``fill`` is mapped or
set; ``alpha`` applies to the fill only, so ``geom_density(aes(fill="g"),
alpha=0.4)`` gives overlapping translucent densities. The default position is
``"identity"`` (``stat_density()`` stacks).

``orientation="y"`` estimates the density of ``y`` and draws it along the y
axis (``x = after_stat("density")``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from manim import VMobject

from ..registry import register
from ..stats.boxplot import flip_names, resolve_flipped
from ..stats.density import density_mapping
from ..theme import Theme, theme_default
from ._rect import filled_key
from .area import GeomArea, _check_outline_type
from .base import make_layer
from .interval import _flip_panel

__all__ = ["GeomDensity", "geom_density"]


def _is_flipped(data: pd.DataFrame, params: Mapping[str, Any]) -> bool:
    return resolve_flipped(data, None, params.get("orientation"), main_is_orthogonal=False)


class GeomDensity(GeomArea):
    """Density areas: one mark per group, like :class:`GeomArea`."""

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

    def setup_data(self, data: pd.DataFrame, params) -> pd.DataFrame:
        flipped = _is_flipped(data, params or {})
        if not flipped:
            return super().setup_data(data, params)
        return flip_names(super().setup_data(flip_names(data), params))

    def draw_panel(
        self, data, panel_params, coord, theme: Theme | None = None, outline_type=None, **params
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        flipped = _is_flipped(data, params)
        data = self.use_defaults(data, params)
        fg = theme.get_foreground()
        unfilled = data["fill"].map(lambda v: v is None or (isinstance(v, float) and v != v))
        data = data.assign(
            alpha=data["alpha"].where(~unfilled, 0.0),
            color=data["color"].map(lambda v: fg if v is None else v),
        )
        params = {k: v for k, v in params.items() if k not in ("alpha", "fill", "color")}
        if flipped:
            data = flip_names(data)
            coord, panel_params = _flip_panel(coord, panel_params)
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
    :mod:`ggmanim.geoms.density` and :mod:`ggmanim.stats.density`.

    Stat params: ``bw`` (``"nrd0"``, ``"nrd"``/``"scott"``, a number or a
    callable), ``adjust``, ``kernel``, ``n`` (512), ``trim``, ``bounds``.
    ``outline_type`` as for :func:`~ggmanim.geoms.area.geom_ribbon`."""
    params["outline_type"] = _check_outline_type(outline_type)
    if stat in ("density", "stat_density"):
        mapping = density_mapping(mapping, params)
    return make_layer(
        GeomDensity(),
        mapping,
        data,
        stat=stat,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
