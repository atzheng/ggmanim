"""geom_col (bars with heights from the data) and geom_bar (bars of counts,
``stat_count``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from manim import VMobject

from ..registry import register
from ..theme import Theme, theme_default
from ._batch import draw_rects_fast, use_fast
from ._rect import draw_rects, filled_key
from .base import Geom, attach_marks, finite_rows, make_layer

__all__ = ["GeomCol", "GeomBar", "geom_col", "geom_bar", "resolution"]


def resolution(x) -> float:
    """Smallest gap between distinct finite values of ``x`` (1 if fewer than two)."""
    x = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce"), dtype=float)
    u = np.unique(x[np.isfinite(x)])
    if len(u) < 2:
        return 1.0
    d = np.diff(u)
    d = d[d > 1e-9 * max(1.0, float(np.abs(u).max()))]
    return float(d.min()) if len(d) else 1.0


class GeomCol(Geom):
    """Bars. ``draw_panel`` returns a ``VGroup`` with one ``VMobject`` per (finite,
    visible) row, in data order (``.marks``/``.keys``: one per drawn row); a
    dashed outline (``linetype``) makes a row's mark ``VGroup(fill, dashes)``.

    :meth:`setup_data` (run by the pipeline after the stat, in transformed space)
    adds ``xmin = x - width/2``, ``xmax = x + width/2``, ``ymin = min(y, 0)``,
    ``ymax = max(y, 0)``. ``width`` (param or column, transformed x units)
    defaults to ``0.9 * resolution(x)`` (``0.9`` on a discrete x scale, where
    levels sit at ``1..n``); so on ``scale_x_log10`` bars have equal
    widths on screen, as in ggplot2. The base ``0`` is in transformed space (data
    value 1 on a log10 y scale, like ggplot2), and both ends are clipped to the
    panel (``clip="on"``) so bars never extend past the axes.

    Drawing and styling are shared with ``geom_rect``/``geom_tile``
    (:func:`~._rect.draw_rects`): the four corners go through
    ``coord.transform``, so bars are correct under any scale transform;
    ``color`` (outline) defaults to ``None`` = no outline, and ``alpha``
    applies to the fill only (as in ggplot2). Seams are not sealed by default
    (``seal = False``, like ``GeomRect``; a subclass may set it).

    ``fast`` (``None``: above ``theme.fast_threshold`` rows) draws batched marks,
    one ``VMobject`` per style (:mod:`ggmanim.geoms._batch`; no per-mark
    ``.keys``).
    """

    required_aes = ("x", "y")
    default_aes = {
        "color": None,
        "fill": "#595959",
        "linewidth": 1.0,
        "alpha": 1.0,
        "linetype": "solid",
    }
    extra_params: tuple[str, ...] = ("width", "fast")
    na_aes: tuple[str, ...] | None = ("xmin", "xmax", "ymin", "ymax")
    default_position = "stack"
    #: Hide anti-aliasing seams between abutting bars (see ``_rect.seal_seams``).
    seal = False

    def setup_data(self, data: pd.DataFrame, params) -> pd.DataFrame:
        data = data.copy()
        if len(data) == 0:
            for c in ("xmin", "xmax", "ymin", "ymax"):
                data[c] = pd.Series(dtype=float)
            return data
        x = pd.to_numeric(data["x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(data["y"], errors="coerce").to_numpy(dtype=float)
        width = params.get("width") if params else None
        if width is None and "width" in data.columns:
            width = pd.to_numeric(data["width"], errors="coerce").to_numpy(dtype=float)
        if width is None:
            # discrete x: levels sit at 1..n, so 0.9 of the level spacing
            width = 0.9 * (1.0 if params and params.get("_x_discrete") else resolution(x))
        width = np.broadcast_to(np.asarray(width, dtype=float), x.shape)
        data["xmin"] = x - width / 2
        data["xmax"] = x + width / 2
        data["ymin"] = np.minimum(y, 0.0)
        data["ymax"] = np.maximum(y, 0.0)
        return data

    def draw_panel(
        self,
        data,
        panel_params,
        coord,
        theme: Theme | None = None,
        width=None,
        *,
        fast=None,
        **params,
    ) -> VMobject:
        theme = theme if theme is not None else theme_default()
        self.check_required_aes(data)
        if not {"xmin", "xmax", "ymin", "ymax"} <= set(data.columns):
            data = self.setup_data(data, {"width": width})
        data = finite_rows(self.use_defaults(data, params), ("xmin", "xmax", "ymin", "ymax"))
        if use_fast(fast, data, theme, type(self).__name__):
            return draw_rects_fast(
                data,
                panel_params,
                coord,
                theme,
                lambda rows: self.draw_panel(
                    rows, panel_params, coord, theme, width, fast=False, **params
                ),
                seal=self.seal,
            )
        bars, drawn = draw_rects(data, panel_params, coord, theme, seal=self.seal)
        return attach_marks(bars, drawn, per="row")

    def draw_key(self, data_row, theme: Theme | None = None) -> VMobject:
        theme = theme if theme is not None else theme_default()
        return filled_key(self.key_aes(data_row), theme)


@register
def geom_col(
    mapping=None,
    data=None,
    *,
    stat="identity",
    position=None,
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Bars whose heights are the data's ``y``. ``width=`` in x (transformed) units;
    constant ``fill``/``color``/``alpha`` go in ``params``."""
    return make_layer(
        GeomCol(),
        mapping,
        data,
        stat=stat,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )


class GeomBar(GeomCol):
    """:class:`GeomCol` fed by ``stat_count`` by default (ggplot2's ``GeomBar``)."""

    default_stat = "count"


@register
def geom_bar(
    mapping=None,
    data=None,
    *,
    stat="count",
    position=None,
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Bars of the number of rows at each x (``stat="count"``; a string column
    is counted per level). ``y=after_stat("count")`` by default (y title
    "count"); ``after_stat("prop")`` too; a ``weight`` aesthetic sums weights.
    ``stat="identity"`` gives :func:`geom_col`. Stacked by default."""
    from .count import stat_layer

    return stat_layer(
        stat,
        GeomBar(),
        mapping,
        data,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
