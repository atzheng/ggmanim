"""geom_count (points sized by stat_sum counts) and the layer plumbing behind the
standalone ``stat_*`` constructors (``stat_ecdf``, ``stat_qq``, ``stat_sum`` ...).

``stat_ecdf`` draws with :class:`~ggmanim.geoms.step.GeomStep`. Its padding
rows sit at ``x = +-inf``, which ggplot2's coord squishes to the panel edges;
``GeomStep`` drops non-finite rows, so ``geom="step"`` resolves to
:class:`GeomStepEdges`, a ``GeomStep`` that first moves infinite positions onto
the panel edges.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from manim import VMobject

from ..aes import Aes, AfterStat, standardise_aes_name
from ..registry import register
from ..stats import STATS
from ..theme import Theme
from .base import Geom, make_layer
from .point import GeomPoint
from .step import GeomStep

__all__ = ["GeomCount", "geom_count"]


class GeomCount(GeomPoint):
    """:class:`GeomPoint` fed by ``stat_sum``: one point per distinct position,
    ``size`` mapped to the count ``after_stat("n")`` unless given."""

    default_stat = "sum"


@register
def geom_count(
    mapping=None,
    data=None,
    *,
    stat="sum",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Points sized by the number of observations at each position (ggplot2's
    ``geom_count``). Map ``size=after_stat("prop")`` for proportions within the
    group; a ``weight`` aesthetic sums weights instead of counting."""
    return stat_layer(
        stat,
        GeomCount(),
        mapping,
        data,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )


# -- step geom for stat_ecdf --------------------------------------------------------


class GeomStepEdges(GeomStep):
    """:class:`~ggmanim.geoms.step.GeomStep` that keeps ``x``/``y = +-inf`` rows
    (``stat_ecdf``'s padding) and draws them on the panel edges, like ggplot2's
    coord squishing infinite values. NaN positions are still dropped."""

    def handle_na(
        self, data: pd.DataFrame, params: Mapping[str, Any] | None = None, label: str | None = None
    ) -> pd.DataFrame:
        # +-inf are kept (drawn on the panel edges); only NaN positions are missing
        if not len(data):
            return data
        finite = {
            c: pd.to_numeric(data[c], errors="coerce").replace([np.inf, -np.inf], 0.0)
            for c in ("x", "y")
            if c in data.columns
        }
        kept = super().handle_na(data.assign(**finite), params, label)
        return data.loc[kept.index]

    def draw_panel(
        self,
        data,
        panel_params,
        coord,
        theme: Theme | None = None,
        label=None,
        direction: str = "hv",
        **params,
    ) -> VMobject:
        if len(data):
            data = data.assign(
                x=_edges(data["x"], getattr(panel_params, "x_range", None)),
                y=_edges(data["y"], getattr(panel_params, "y_range", None)),
            )
        return super().draw_panel(
            data, panel_params, coord, theme, label=label, direction=direction, **params
        )


def _edges(values: pd.Series, rng) -> np.ndarray:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if rng is None:
        return v
    lo, hi = float(min(rng[0], rng[1])), float(max(rng[0], rng[1]))
    return np.where(v == -np.inf, lo, np.where(v == np.inf, hi, v))


# -- stat layers --------------------------------------------------------------------


def _resolve_geom(geom) -> Geom:
    """A ``Geom`` instance from an instance, a class or a name (``"point"``,
    ``"qq_line"``; ``"step"`` -> :class:`GeomStepEdges`)."""
    if isinstance(geom, Geom):
        return geom
    if isinstance(geom, type) and issubclass(geom, Geom):
        return geom()
    if isinstance(geom, str):
        stem = geom[5:] if geom.startswith("geom_") else geom
        if stem == "step":
            return GeomStepEdges()
        from .. import geoms

        cls = getattr(geoms, "Geom" + "".join(p.capitalize() for p in stem.split("_")), None)
        if isinstance(cls, type) and issubclass(cls, Geom):
            return cls()
    raise ValueError(f"Cannot interpret {geom!r} as a geom")


def _stat_default_mapping(stat, mapping, params: Mapping[str, Any]) -> Aes | None:
    """``mapping`` plus the stat's ``after_stat`` defaults (e.g. ``stat_sum``'s
    ``size=after_stat("n")``) for aesthetics neither mapped nor set as constants."""
    cls = (
        STATS.get(stat[5:] if stat.startswith("stat_") else stat)
        if isinstance(stat, str)
        else (stat if isinstance(stat, type) else type(stat))
    )
    defaults = {
        k: v
        for k, v in (getattr(cls, "default_aes", None) or {}).items()
        if isinstance(v, AfterStat)
    }
    if not defaults:
        return mapping
    given = Aes(mapping) if mapping is not None and not isinstance(mapping, pd.DataFrame) else Aes()
    set_ = {standardise_aes_name(k) for k in params}
    extra = {k: v for k, v in defaults.items() if k not in given and k not in set_}
    return Aes(extra).merge(given) if extra else mapping


def stat_layer(
    stat,
    geom,
    mapping=None,
    data=None,
    *,
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    params: Mapping[str, Any] | None = None,
):
    """The layer behind a standalone ``stat_<stat>(...)`` constructor: ``geom``
    (instance, class or name) fed by ``stat``, with the stat's ``after_stat``
    default aesthetics added where not mapped or set."""
    params = dict(params or {})
    if isinstance(mapping, pd.DataFrame) and (data is None or isinstance(data, Mapping)):
        mapping, data = data, mapping
    return make_layer(
        _resolve_geom(geom),
        _stat_default_mapping(stat, mapping, params),
        data,
        stat=stat,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
