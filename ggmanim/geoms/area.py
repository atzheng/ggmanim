"""geom_ribbon and geom_area: filled bands between ``ymin`` and ``ymax``.

* :func:`geom_ribbon` -- the band between ``aes(ymin=, ymax=)`` along ``x``.
* :func:`geom_area` -- a ribbon from 0 to ``y`` (``setup_data`` sets ``ymin = 0``,
  ``ymax = y``), stacked by default (``position="stack"``).

Each group is sorted by ``x`` and drawn as one filled ``VMobject`` (clipped to the
panel with :func:`~._clip.clip_polygon`); a row with a missing ``x``/``ymin``/
``ymax`` breaks the band into separate pieces (subpaths of the same mobject), as
in ggplot2. ``outline_type`` picks which edges get the ``color`` outline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from manim import VGroup, VMobject

from ..positions import POSITIONS, Position, get_position
from ..registry import register
from ..theme import Theme, theme_default
from ._clip import clip_polygon, panel_rect
from ._linetype import apply_linetype
from ._rect import filled_key, rings_mobject, style_filled
from .base import (
    Geom,
    attach_marks,
    clipped_path_points,
    first,
    make_layer,
    multi_path_mobject,
    num,
    resolve_color,
    transform_points,
)

__all__ = ["GeomRibbon", "GeomArea", "geom_ribbon", "geom_area"]

OUTLINE_TYPES = ("both", "upper", "lower", "full")


def _check_outline_type(outline_type: Any) -> str:
    if outline_type not in OUTLINE_TYPES:
        raise ValueError(
            f"outline_type must be one of {', '.join(map(repr, OUTLINE_TYPES))}, "
            f"got {outline_type!r}"
        )
    return outline_type


def _runs(ok: np.ndarray) -> list[slice]:
    """Maximal runs of ``True`` in ``ok`` with at least two elements."""
    out = []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(int), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= 2:
            out.append(slice(int(a), int(b)))
    return out


class GeomRibbon(Geom):
    """Bands between ``ymin`` and ``ymax``: one mark per group (in ``group``
    order).

    A group's mark is its filled ``VMobject`` (one subpath per run of finite
    rows), or ``VGroup(fill, outline)`` when ``color`` is set; the outline follows
    ``outline_type``: ``"both"`` (default: upper and lower edges, not the ends),
    ``"upper"``, ``"lower"`` or ``"full"`` (the closed polygon). ``alpha`` applies
    to the fill only; ``color=None`` (default) means no outline.
    """

    required_aes: tuple[str, ...] = ("x", "ymin", "ymax")
    default_aes: dict[str, Any] = {
        "color": None,
        "fill": "#595959",
        "linewidth": 3.0,
        "alpha": 1.0,
        "linetype": "solid",
    }
    extra_params: tuple[str, ...] = ("outline_type",)
    default_outline_type = "both"

    def handle_na(self, data, params=None, label=None) -> pd.DataFrame:
        """Missing values are kept: they break the band into pieces when drawn."""
        return data

    def _bounds(self, gdf: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return gdf["ymin"].to_numpy(float), gdf["ymax"].to_numpy(float)

    def draw_panel(
        self, data, panel_params, coord, theme: Theme | None = None, outline_type=None, **params
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        outline_type = _check_outline_type(
            self.default_outline_type if outline_type is None else outline_type
        )
        self.check_required_aes(data)
        data = self.use_defaults(data, params)
        rect = panel_rect(coord, panel_params)
        out = VGroup()
        drawn_groups = []
        for gid, gdf in data.groupby("group", sort=True):
            gdf = gdf.sort_values("x", kind="stable")
            x = pd.to_numeric(gdf["x"], errors="coerce").to_numpy(float)
            lo, hi = self._bounds(gdf)
            ok = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
            rings, lines = [], []
            for s in _runs(ok):
                xs, los, his = x[s], lo[s], hi[s]
                rx = np.concatenate([xs, xs[::-1]])
                ry = np.concatenate([his, los[::-1]])
                cx, cy = (rx, ry) if rect is None else clip_polygon(rx, ry, rect)
                if len(cx) >= 3:
                    rings.append(transform_points(coord, cx, cy))
                if outline_type in ("both", "upper"):
                    lines.append((xs, his))
                if outline_type in ("both", "lower"):
                    lines.append((xs, los))
                if outline_type == "full":
                    lines.append((np.append(rx, rx[0]), np.append(ry, ry[0])))
            fill = style_filled(
                rings_mobject(rings),
                {"fill": first(gdf, "fill"), "alpha": first(gdf, "alpha"), "color": None},
                theme,
            )
            mob: VMobject = fill
            color = resolve_color(first(gdf, "color"))
            pieces = []
            if color is not None:
                for lx, ly in lines:
                    pieces.extend(clipped_path_points(coord, panel_params, lx, ly))
            if pieces:
                outline = multi_path_mobject(pieces, color, num(first(gdf, "linewidth"), 3.0), 1.0)
                mob = VGroup(fill, apply_linetype(outline, first(gdf, "linetype")))
            elif not rings:
                continue
            out.add(mob)
            drawn_groups.append(gid)
        drawn = data[data["group"].isin(drawn_groups)]
        return attach_marks(out, drawn, per="group")

    def draw_key(self, data_row, theme: Theme | None = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        return filled_key(self.key_aes(data_row), theme, default_linewidth=3.0)


class _DeferredPosition(Position):
    """A named position looked up with :func:`~ggmanim.positions.get_position` only
    when the build runs it, so ``geom_area()`` (default ``position="stack"``) can
    be created before that position is registered. Delegates every pipeline hook
    to the resolved position. (No ``name`` in the class body: not registered.)"""

    def __init__(self, target: str, **params: Any) -> None:
        super().__init__(**params)
        self.target = target

    def __repr__(self) -> str:
        return f"position_{self.target}(deferred)"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _DeferredPosition)
            and other.target == self.target
            and other.params == self.params
        )

    __hash__ = None  # type: ignore[assignment]

    def resolve(self) -> Position:
        return get_position(self.target, **self.params)

    def setup_params(self, data):
        return self.resolve().setup_params(data)

    def setup_data(self, data, params):
        return self.resolve().setup_data(data, params)

    def compute_layer(self, data, params, scales=None):
        return self.resolve().compute_layer(data, params, scales)


def _position(position: Any, default: str) -> Any:
    """``position``, or a :class:`_DeferredPosition` when it is the geom's default
    name and that position is not registered (yet)."""
    if isinstance(position, str):
        key = position[9:] if position.startswith("position_") else position
        if key == default and key not in POSITIONS:
            return _DeferredPosition(key)
    return position


class GeomArea(GeomRibbon):
    """Areas from 0 to ``y``: :meth:`setup_data` (transformed space) adds
    ``ymin = 0`` and ``ymax = y``, which the default ``position="stack"``
    stacks. Drawn like :class:`GeomRibbon` with ``outline_type="upper"``."""

    required_aes = ("x", "y")
    default_position = "stack"
    default_outline_type = "upper"

    def setup_data(self, data: pd.DataFrame, params) -> pd.DataFrame:
        data = data.copy()
        y = data["y"] if "y" in data.columns else pd.Series(np.nan, index=data.index)
        data["ymin"] = np.zeros(len(data))
        data["ymax"] = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
        return data

    def _bounds(self, gdf: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if {"ymin", "ymax"} <= set(gdf.columns):
            return super()._bounds(gdf)
        y = pd.to_numeric(gdf["y"], errors="coerce").to_numpy(float)
        return np.zeros(len(y)), y


@register
def geom_ribbon(
    mapping=None,
    data=None,
    *,
    outline_type: str = "both",
    stat="identity",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Ribbons between ``aes(ymin=, ymax=)`` along ``x`` (one per group).

    ``outline_type`` (``"both"``, ``"upper"``, ``"lower"``, ``"full"``) picks the
    edges outlined in ``color`` (no outline by default); ``alpha`` is the fill's."""
    params["outline_type"] = _check_outline_type(outline_type)
    return make_layer(
        GeomRibbon(),
        mapping,
        data,
        stat=stat,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )


@register
def geom_area(
    mapping=None,
    data=None,
    *,
    outline_type: str = "upper",
    stat="identity",
    position="stack",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Filled areas under ``aes(x, y)`` (from 0), stacked by group by default.

    ``position="identity"`` overlaps the areas instead of stacking them. The
    ``"stack"`` position is resolved when the plot is built. ``outline_type``
    as for :func:`geom_ribbon` (default ``"upper"``)."""
    params["outline_type"] = _check_outline_type(outline_type)
    return make_layer(
        GeomArea(),
        mapping,
        data,
        stat=stat,
        position=_position(
            GeomArea.default_position if position is None else position, GeomArea.default_position
        ),
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
