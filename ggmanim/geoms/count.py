"""geom_count (points sized by stat_sum counts) and the layer plumbing behind the
standalone ``stat_*`` constructors (``stat_ecdf``, ``stat_qq``, ``stat_sum`` ...).

``stat_ecdf`` draws with a step function. Until ``geom_step`` exists (bead
ggm-2m8.3), :class:`GeomStepFallback` -- a :class:`GeomPath` that expands the
data into a staircase and resolves ``x = +-inf`` to the panel edges -- stands in
for it; ``geom="step"`` resolves to it.
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
from .base import Geom, finite_rows, make_layer
from .line import GeomPath
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


# -- step fallback ----------------------------------------------------------------


def _step_rows(n: int, direction: str) -> tuple[np.ndarray, np.ndarray]:
    """Source rows of the staircase vertices through ``n`` points sorted by x: the
    row giving each vertex's x and the row giving its y (and other columns)."""
    if direction not in ("hv", "vh", "mid"):
        raise ValueError(f"step direction must be 'hv', 'vh' or 'mid', got {direction!r}")
    if n < 2:
        return np.arange(n), np.arange(n)
    i = np.repeat(np.arange(n), 2)
    if direction == "hv":  # across, then up
        return i[1:], i[:-1]
    if direction == "vh":  # up, then across
        return i[:-1], i[1:]
    return i, i  # mid: x is replaced by the midpoints


def step_expand(x, y, direction: str = "hv") -> tuple[np.ndarray, np.ndarray]:
    """Staircase vertices ``(xs, ys)`` through ``(x, y)`` (sorted by x): ``"hv"``
    goes across then up, ``"vh"`` up then across, ``"mid"`` steps half-way
    between consecutive x values."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xr, yr = _step_rows(len(x), direction)
    xs = x[xr]
    if direction == "mid" and len(x) > 1:
        mids = (x[:-1] + x[1:]) / 2
        xs = np.concatenate([[x[0]], np.repeat(mids, 2), [x[-1]]])
    return xs, y[yr]


class GeomStepFallback(GeomPath):
    """A step function drawn as a :class:`GeomPath` (stand-in for ``geom_step``).

    Each group is sorted by x and expanded into a staircase (``direction``
    ``"hv"``/``"vh"``/``"mid"``). Infinite x / y (``stat_ecdf``'s padding) are
    placed on the panel edges; NaN rows are dropped.
    """

    extra_params: tuple[str, ...] = ("label", "direction")

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
        self.check_required_aes(data)
        pieces = []
        for _, gdf in data.groupby("group", sort=True):
            gdf = gdf.sort_values("x", kind="stable")
            gdf = gdf.assign(
                x=_edges(gdf["x"], getattr(panel_params, "x_range", None)),
                y=_edges(gdf["y"], getattr(panel_params, "y_range", None)),
            )
            gdf = finite_rows(gdf, ("x", "y"))
            _, rows = _step_rows(len(gdf), direction)
            xs, ys = step_expand(gdf["x"], gdf["y"], direction)
            pieces.append(gdf.iloc[rows].assign(x=xs, y=ys))
        expanded = pd.concat(pieces, ignore_index=True) if pieces else data.iloc[0:0]
        return super().draw_panel(expanded, panel_params, coord, theme, label=label, **params)


def _edges(values: pd.Series, rng) -> np.ndarray:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if rng is None:
        return v
    lo, hi = float(min(rng[0], rng[1])), float(max(rng[0], rng[1]))
    return np.where(v == -np.inf, lo, np.where(v == np.inf, hi, v))


# -- stat layers --------------------------------------------------------------------


def _resolve_geom(geom) -> Geom:
    """A ``Geom`` instance from an instance, a class or a name (``"point"``,
    ``"qq_line"``; ``"step"`` -> :class:`GeomStepFallback`)."""
    if isinstance(geom, Geom):
        return geom
    if isinstance(geom, type) and issubclass(geom, Geom):
        return geom()
    if isinstance(geom, str):
        stem = geom[5:] if geom.startswith("geom_") else geom
        if stem == "step":  # TODO(ggm-2m8.3): use geoms.step.GeomStep once merged
            return GeomStepFallback()
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
