"""geom_count (points sized by stat_sum counts) and the layer plumbing behind the
standalone ``stat_*`` constructors (``stat_ecdf``, ``stat_qq``, ``stat_sum`` ...).

``stat_ecdf`` draws with :class:`~ggmanim.geoms.step.GeomStep`. Its padding
rows sit at ``x = +-inf``, which the path geoms draw on the panel edges (like
ggplot2's coord squishing infinite values).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from ..registry import register
from .base import Geom, make_layer
from .point import GeomPoint

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


# -- stat layers --------------------------------------------------------------------


def _resolve_geom(geom) -> Geom:
    """A ``Geom`` instance from an instance, a class or a name (``"point"``,
    ``"qq_line"``)."""
    if isinstance(geom, Geom):
        return geom
    if isinstance(geom, type) and issubclass(geom, Geom):
        return geom()
    if isinstance(geom, str):
        stem = geom[5:] if geom.startswith("geom_") else geom
        from .. import geoms

        cls = getattr(geoms, "Geom" + "".join(p.capitalize() for p in stem.split("_")), None)
        if isinstance(cls, type) and issubclass(cls, Geom):
            return cls()
    raise ValueError(f"Cannot interpret {geom!r} as a geom")


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
    (instance, class or name) fed by ``stat`` (:func:`make_layer` adds the stat's
    ``after_stat`` default aesthetics where not mapped or set)."""
    params = dict(params or {})
    if isinstance(mapping, pd.DataFrame) and (data is None or isinstance(data, Mapping)):
        mapping, data = data, mapping
    return make_layer(
        _resolve_geom(geom),
        mapping,
        data,
        stat=stat,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
