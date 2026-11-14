"""The :class:`Facet` contract: how a facet splits a plot into panels.

A facet is a plot component (``plot + facet_wrap("class")``). The build pipeline
(:mod:`ggmanim.build`) asks it four things; ``facet_wrap`` / ``facet_grid`` only
need to implement :meth:`Facet.compute_layout` and :meth:`Facet.vars` (plus
:meth:`Facet.strips` for their strip labels) -- everything else has a generic
implementation here, mirroring ggplot2's ``Facet`` ggproto.

Pipeline
--------
1. **Layout.** ``layout = facet.compute_layout([plot.data, *layer data])`` --
   the *layout table*, one row per panel:

   ==========  ===============================================================
   ``PANEL``   panel id, ``1..n`` (int), in drawing order; panel 1 is the one
               ``GGPlot.panel`` / ``GGPlot.axes`` refer to
   ``ROW``     1-based grid row (top = 1)
   ``COL``     1-based grid column (left = 1)
   ``SCALE_X`` which x scale the panel uses (int id); panels with equal ids
               share one trained scale object
   ``SCALE_Y`` the same for y
   *vars*      one column per facet variable (:meth:`vars`): the panel's value
   ==========  ===============================================================

   ``data`` holds the *raw* user data frames (``None`` entries removed): the plot
   data and every layer's own data, each first passed through
   :meth:`Facet.facet_data` (which may add computed facet columns, e.g.
   ``facet_wrap({"big": fn})``). A facet variable missing from every frame is
   an error (raise :class:`ValueError`). Use :meth:`scale_ids` to fill
   ``SCALE_X``/``SCALE_Y`` from the ``scales=`` argument.

2. **Rows -> panels.** ``data = facet.map_data(data, layout)`` per layer: ``data``
   is the layer's evaluated aesthetics (``x``, ``y``, ``colour`` ..., ``group``)
   *plus* the facet-variable columns of the layer's raw data that exist there.
   It returns the rows with an int ``PANEL`` column. The generic version matches
   rows to panels on the facet variables present; a layer lacking some (or all)
   variables is *repeated* in every panel matching the ones it has (ggplot2's
   rule, which also puts ``geom_vline``/``geom_function`` in every panel).
   Layers with no rows at all (``geom_function``) are computed once per panel by
   the pipeline.

3. **Scales.** ``panel_scales = facet.init_scales(layout, x_scale, y_scale)``
   -> ``{PANEL: {"x": scale, "y": scale}}``: the given (untrained) scales when
   every panel has the same ``SCALE_X``/``SCALE_Y`` (``scales="fixed"``), else one
   ``scale.clone()`` per distinct id. The pipeline trains each panel's scales on
   that panel's rows only (pre-stat, then again after positions) and runs stats
   and positions per panel with that panel's scales, so free scales train
   independently. Non-position scales (colour, size ...) are always shared.
   :meth:`finish_data` is a last per-layer hook after the position scales are
   trained (default: identity).

4. **Panels.** The pipeline builds one :class:`~ggmanim.coords.CartesianPanel`
   per layout row with ``coord.build`` and that panel's scales, sized so the
   whole grid fits :meth:`Theme.panel_size <ggmanim.theme.Theme.panel_size>`,
   with tick marks/labels only where :meth:`draw_axes` says, and axis titles
   suppressed (drawn once for the grid). :meth:`draw_panels` then adds the
   :meth:`strips` and places every panel in its cell, returning a
   :class:`PanelGrid`. Override :meth:`panel_weights` for unequal column widths /
   row heights (``facet_grid(space="free")``).

Result
------
``GGPlot.panels[(row, col)]`` is each panel; ``GGPlot.panel`` / ``axes`` the
first; ``GGPlot.layers[name]`` a :class:`PanelLayers` (one ``VGroup`` per panel,
``.panel(row, col)``, ``.marks`` flattened in panel order); ``GGPlot.strips`` a
``VGroup`` of every strip. With :class:`~ggmanim.facets.null.FacetNull` (no
facet) the built plot is exactly the single-panel plot.

Writing a facet
---------------
::

    class FacetWrap(Facet):
        def __init__(self, facets, scales="fixed", labeller=None):
            super().__init__(scales=scales, labeller=labeller)
            self.facets = [facets] if isinstance(facets, str) else list(facets)

        def vars(self):
            return list(self.facets)

        def compute_layout(self, data):
            levels = self.combinations(data)      # DataFrame of observed combos
            n = len(levels)
            ncol = math.ceil(math.sqrt(n))
            layout = levels.assign(PANEL=range(1, n + 1),
                                   ROW=[i // ncol + 1 for i in range(n)],
                                   COL=[i % ncol + 1 for i in range(n)])
            return self.scale_ids(layout)

        def strips(self, layout):
            return {p: [("top", self.strip_labels(layout, p, self.facets))]
                    for p in layout["PANEL"]}
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from manim import DOWN, LEFT, RIGHT, UP, Mobject, VGroup

from ..theme import Theme
from . import _layout, _strips
from .labellers import as_labeller

__all__ = ["Facet", "PanelGrid", "PanelLayers", "LAYOUT_COLUMNS"]

#: Columns every layout table has (plus one per facet variable).
LAYOUT_COLUMNS = ("PANEL", "ROW", "COL", "SCALE_X", "SCALE_Y")

_SCALES = ("fixed", "free", "free_x", "free_y")
_PLACEMENTS = ("inside", "outside")

StripSpecs = dict[int, list[tuple[str, list[str]]]]


class Facet:
    """Base facet. See the module docstring for the contract.

    Parameters
    ----------
    scales : ``"fixed"`` (all panels share x and y), ``"free_x"``, ``"free_y"`` or
        ``"free"`` (each panel trains its own x and/or y scale).
    labeller : strip labeller (:mod:`ggmanim.facets.labellers`); ``None`` =
        :func:`~ggmanim.facets.labellers.label_value`.
    """

    #: ``True`` only for :class:`~ggmanim.facets.null.FacetNull`: the pipeline then
    #: builds the classic single panel (no strips, axis titles on the panel).
    is_null: ClassVar[bool] = False

    def __init__(self, scales: str = "fixed", labeller: Any = None):
        if scales not in _SCALES:
            raise ValueError(f"scales must be one of {_SCALES}, got {scales!r}")
        self.scales = scales
        self.labeller = as_labeller(labeller)

    def add_to(self, plot):
        return plot._with_facet(self)

    def __repr__(self):
        return f"{type(self).__name__}(scales={self.scales!r})"

    @property
    def free(self) -> dict[str, bool]:
        """``{"x": bool, "y": bool}``: which position scales vary by panel."""
        return {
            "x": self.scales in ("free", "free_x"),
            "y": self.scales in ("free", "free_y"),
        }

    # -- 1. layout -------------------------------------------------------------
    def vars(self) -> list[str]:
        """Names of the facet variables (data columns), outermost first."""
        return []

    def facet_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Hook run by the build on the plot data and on every layer's raw data
        before :meth:`compute_layout`: return ``data`` plus any computed facet
        columns (never modify it in place). Default: ``data`` unchanged."""
        return data

    def compute_layout(self, data: Sequence[pd.DataFrame]) -> pd.DataFrame:
        """The layout table (see the module docstring). Must be overridden."""
        raise NotImplementedError(f"{type(self).__name__}.compute_layout")

    def combinations(self, data: Sequence[pd.DataFrame], vars: Sequence[str] | None = None):
        """Observed combinations of ``vars`` (default :meth:`vars`) over every
        frame that has *all* of them, as a DataFrame sorted by each variable's
        levels (categorical order, else sorted values; missing values last).
        Raises :class:`ValueError` if a variable is in no frame."""
        vars = list(self.vars() if vars is None else vars)
        if not vars:
            return pd.DataFrame(index=range(1))
        for v in vars:
            if not any(v in df.columns for df in data):
                raise ValueError(f"facet variable {v!r} is not a column of the plot or layer data")
        frames = [df[vars] for df in data if all(v in df.columns for v in vars)]
        if not frames:
            # each variable lives in a different frame: cross the observed levels
            levels = [_levels([df[v] for df in data if v in df.columns]) for v in vars]
            idx = pd.MultiIndex.from_product(levels, names=vars)
            return idx.to_frame(index=False)
        combos = pd.concat(frames, ignore_index=True).drop_duplicates()
        order = []
        for v in vars:
            levels = _levels([df[v] for df in data if v in df.columns])
            order.append(combos[v].map(_position_of(levels)))
        keys = pd.concat(order, axis=1, keys=range(len(vars)))
        combos = combos.iloc[np.lexsort(keys.to_numpy().T[::-1])]
        return combos.reset_index(drop=True)

    def scale_ids(
        self, layout: pd.DataFrame, x_by: str = "PANEL", y_by: str = "PANEL"
    ) -> pd.DataFrame:
        """``layout`` with ``SCALE_X``/``SCALE_Y``: the ``x_by``/``y_by`` column
        (``"PANEL"``; ``facet_grid`` uses ``"COL"``/``"ROW"``) when that scale is
        free, else 1."""
        free = self.free
        return layout.assign(
            SCALE_X=layout[x_by].astype(int) if free["x"] else 1,
            SCALE_Y=layout[y_by].astype(int) if free["y"] else 1,
        )

    # -- 2. rows -> panels -----------------------------------------------------
    def map_data(self, data: pd.DataFrame, layout: pd.DataFrame) -> pd.DataFrame:
        """``data`` with a ``PANEL`` column (rows repeated across panels for facet
        variables ``data`` lacks); row order is kept within each panel."""
        if not len(data):
            return data
        present = [v for v in self.vars() if v in data.columns and v in layout.columns]
        panels = layout["PANEL"].astype(int).to_numpy()
        if not present:
            pieces = [data.assign(PANEL=p) for p in panels]
            return pd.concat(pieces, ignore_index=True)
        # integer codes per variable (robust to dtype differences, NaN-safe)
        left = pd.DataFrame({"_row": np.arange(len(data))})
        right = pd.DataFrame({"PANEL": panels})
        for k, v in enumerate(present):
            levels = pd.Index(pd.unique(layout[v]))
            left[f"_k{k}"] = levels.get_indexer(data[v])
            right[f"_k{k}"] = levels.get_indexer(layout[v])
        keys = [f"_k{k}" for k in range(len(present))]
        matched = left.merge(right, on=keys, how="inner").sort_values(
            ["PANEL", "_row"], kind="stable"
        )
        out = data.iloc[matched["_row"].to_numpy()].assign(
            PANEL=matched["PANEL"].to_numpy().astype(int)
        )
        return out.reset_index(drop=True)

    # -- 3. scales -------------------------------------------------------------
    def init_scales(self, layout: pd.DataFrame, x_scale: Any, y_scale: Any) -> dict[int, dict]:
        """``{PANEL: {"x": scale, "y": scale}}``: ``x_scale``/``y_scale`` themselves
        when one id is used, else one clone per ``SCALE_X``/``SCALE_Y`` id."""

        def per_id(col: str, proto: Any) -> dict[int, Any]:
            ids = sorted(int(i) for i in pd.unique(layout[col]))
            if len(ids) == 1:
                return {ids[0]: proto}
            return {i: proto.clone() for i in ids}

        xs, ys = per_id("SCALE_X", x_scale), per_id("SCALE_Y", y_scale)
        return {
            int(r.PANEL): {"x": xs[int(r.SCALE_X)], "y": ys[int(r.SCALE_Y)]}
            for r in layout.itertuples(index=False)
        }

    def finish_data(self, data: pd.DataFrame, layout: pd.DataFrame) -> pd.DataFrame:
        """Last per-layer hook once position scales are trained. Default: as is."""
        return data

    # -- 4. panels -------------------------------------------------------------
    def strips(self, layout: pd.DataFrame) -> StripSpecs:
        """``{PANEL: [(side, labels), ...]}``: the strips of each panel. ``side`` is
        ``"top"``, ``"bottom"``, ``"left"`` or ``"right"``; ``labels`` lists one
        string per stacked strip, outermost first (:meth:`strip_labels` makes it
        with the labeller). Default: no strips."""
        return {}

    def strip_labels(self, layout: pd.DataFrame, panel: int, vars: Iterable[str]) -> list[str]:
        """The labeller's strings for ``vars`` of ``panel``'s layout row."""
        row = layout.loc[layout["PANEL"] == panel].iloc[0]
        return [str(s) for s in self.labeller({v: row[v] for v in vars})]

    def draw_axes(
        self,
        layout: pd.DataFrame,
        h_col: str = "SCALE_X",
        v_col: str = "SCALE_Y",
        h_side: str = "bottom",
        v_side: str = "left",
    ) -> dict[int, tuple[bool, bool]]:
        """``{PANEL: (horizontal axis drawn, vertical axis drawn)}``; ``h_col`` is
        the layout column of the horizontal scale (``"SCALE_Y"`` under
        ``coord_flip``). Default: :func:`ggmanim.facets._layout.axis_visibility`
        (outer panels, or every panel whose neighbour uses another scale)."""
        return _layout.axis_visibility(layout, h_col, v_col, h_side, v_side)

    def panel_weights(
        self, layout: pd.DataFrame, panel_params: Mapping[int, Any]
    ) -> tuple[list[float] | None, list[float] | None]:
        """Relative ``(column widths, row heights)`` (``None`` = equal).
        ``panel_params`` is ``{PANEL: PanelParams}`` of a first build."""
        return None, None

    def draw_panels(
        self,
        panels: Mapping[int, Any],
        layout: pd.DataFrame,
        theme: Theme,
        widths: Sequence[float],
        heights: Sequence[float],
        axes: Mapping[int, tuple[bool, bool]] | None = None,
        sides: tuple[str, str] = ("bottom", "left"),
    ) -> PanelGrid:
        """Add the strips to the built ``panels`` (``{PANEL: CartesianPanel}``) and
        place them in a grid of ``widths`` x ``heights`` plotting areas.

        ``axes`` (:meth:`draw_axes`) and ``sides`` (horizontal, vertical axis side)
        decide where strips sharing a side with an axis go: next to the panel
        with the axis moved out (``theme.strip_placement="inside"``) or beyond the
        axis (``"outside"``)."""
        specs = self.strips(layout)
        thickness = strip_thickness(theme, specs)
        units, strips = {}, {}
        for p, panel in panels.items():
            vis = (axes or {}).get(p, (True, True))
            strips[p] = add_strips(panel, specs.get(p, []), thickness, theme, vis, sides)
            units[p] = VGroup(panel, strips[p])
        ext = {p: _layout.extents(u, panels[p].axes) for p, u in units.items()}
        spacing = panel_spacing(theme)
        _layout.place_units(layout, units, dict(panels), ext, widths, heights, spacing)
        return PanelGrid(layout, panels, strips)


# --------------------------------------------------------------------------- helpers
def _levels(series: Sequence[pd.Series]) -> list:
    """Ordered levels of a variable over several columns: categorical
    categories (first categorical wins), else sorted unique values; NaN last."""
    for s in series:
        if isinstance(s.dtype, pd.CategoricalDtype):
            cats = list(s.cat.categories)
            seen = pd.unique(pd.concat([x.astype(object) for x in series], ignore_index=True))
            extra = [v for v in seen if not _isna(v) and v not in cats]
            return cats + extra + ([np.nan] if any(_isna(v) for v in seen) else [])
    values = pd.unique(pd.concat([s.astype(object) for s in series], ignore_index=True))
    present = [v for v in values if not _isna(v)]
    try:
        present = sorted(present)
    except TypeError:
        present = sorted(present, key=str)
    return present + ([np.nan] if len(present) < len(values) else [])


def _isna(v: Any) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _position_of(levels: list):
    index = pd.Index(levels)

    def pos(v):
        i = index.get_indexer([v])[0]
        return i if i >= 0 else len(levels)

    return pos


def panel_spacing(theme: Theme) -> tuple[float, float]:
    """``(horizontal, vertical)`` gap between panels (``panel_spacing_x``/``_y``,
    falling back to ``panel_spacing``)."""
    base = float(getattr(theme, "panel_spacing", 0.3))
    sx = getattr(theme, "panel_spacing_x", None)
    sy = getattr(theme, "panel_spacing_y", None)
    return (base if sx is None else float(sx)), (base if sy is None else float(sy))


def strip_thickness(theme: Theme, specs: StripSpecs) -> float:
    """Common thickness of every strip in ``specs`` (0 when there are none)."""
    labels = [s for entries in specs.values() for _, ls in entries for s in ls]
    return _strips.strip_thickness(theme, labels) if labels else 0.0


def strip_extents(specs: StripSpecs, thickness: float) -> dict[int, dict[str, float]]:
    """``{PANEL: {side: total strip thickness}}``."""
    out: dict[int, dict[str, float]] = {}
    for p, entries in specs.items():
        for side, labels in entries:
            out.setdefault(int(p), {})
            out[int(p)][side] = out[int(p)].get(side, 0.0) + len(labels) * thickness
    return out


def _axis_parts(panel: Any, side: str, sides: tuple[str, str]) -> list:
    """The panel's tick marks + labels on ``side`` (if that is an axis side)."""
    if side == sides[0]:
        return [panel.x_ticks, panel.x_tick_labels]
    if side == sides[1]:
        return [panel.y_ticks, panel.y_tick_labels]
    # a secondary axis (sec_axis) on the opposite side; emptied where not drawn
    return list(getattr(panel, "secondary_parts", {}).get(side, []))


def add_strips(
    panel: Any,
    entries: Sequence[tuple[str, Sequence[str]]],
    thickness: float,
    theme: Theme,
    visible: tuple[bool, bool] = (True, True),
    sides: tuple[str, str] = ("bottom", "left"),
) -> VGroup:
    """Strips for one panel (``entries`` from :meth:`Facet.strips`) against its
    plotting area. Where a strip shares a side with a drawn axis,
    ``theme.strip_placement`` decides: ``"inside"`` (default) shifts the panel's
    tick marks/labels on that side outward past the strips; ``"outside"`` puts
    the strips beyond them."""
    placement = getattr(theme, "strip_placement", "inside")
    if placement not in _PLACEMENTS:
        raise ValueError(f"strip_placement must be one of {_PLACEMENTS}, got {placement!r}")
    offsets: dict[str, float] = {}
    lo, hi = _layout.area_box(panel)
    out_dir = {"top": UP, "bottom": DOWN, "left": LEFT, "right": RIGHT}
    for side, labels in entries:
        parts = _axis_parts(panel, side, sides)
        drawn = visible[0] if side == sides[0] else visible[1] if side == sides[1] else True
        parts = [m for m in parts if len(m)] if drawn else []
        if not parts:
            continue
        if placement == "inside":
            for m in parts:
                m.shift(len(labels) * thickness * out_dir[side])
        else:
            far = VGroup(*parts)
            offsets[side] = {
                "top": far.get_top()[1] - hi[1],
                "bottom": lo[1] - far.get_bottom()[1],
                "left": lo[0] - far.get_left()[0],
                "right": far.get_right()[0] - hi[0],
            }[side] + float(getattr(theme, "strip_margin", 0.08))
    return _strips.attach_strips(panel.axes, entries, thickness, theme, offsets)


# --------------------------------------------------------------------------- mobjects
class PanelGrid(VGroup):
    """Every panel of a faceted plot, their strips and the shared axis titles.

    Attributes
    ----------
    layout : the facet's layout table
    panels : ``{(row, col): CartesianPanel}`` in ``PANEL`` order
    panel_ids : ``{PANEL: CartesianPanel}``
    strips : flat ``VGroup`` of every strip, in panel order (each strip has
        ``.text``, ``.background`` and ``.side``); ``panel_strips[(row, col)]``
        is a ``VGroup`` of one panel's strips (an index, not a submobject)
    axes : ``VGroup`` of the panels' axes (the whole plotting area; not a
        submobject, used for legend placement / title alignment)
    x_title, y_title : the horizontal / vertical axis titles (or ``None``)
    panel_params : the first panel's ``PanelParams``

    Submobjects: the panels (``PANEL`` order), then ``strips``, then the titles.
    """

    def __init__(
        self,
        layout: pd.DataFrame,
        panels: Mapping[int, Any],
        strips: Mapping[int, VGroup] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layout = layout
        cells = {int(r.PANEL): (int(r.ROW), int(r.COL)) for r in layout.itertuples(index=False)}
        order = sorted(panels)
        self.panel_ids = {p: panels[p] for p in order}
        self.panels = {cells[p]: panels[p] for p in order}
        strips = strips or {}
        self.panel_strips = {cells[p]: strips.get(p, VGroup()) for p in order}
        # flat: each strip once, in panel order (panel_strips only index them)
        self.strips = VGroup(*(s for group in self.panel_strips.values() for s in group))
        self.axes = VGroup(*(panels[p].axes for p in order))
        self.panel_params = panels[order[0]].panel_params
        self.x_title: Mobject | None = None
        self.y_title: Mobject | None = None
        self.add(*self.panel_ids.values(), self.strips)

    def panel(self, row: int, col: int):
        """The panel in grid cell ``(row, col)`` (1-based)."""
        return self.panels[(row, col)]

    def add_titles(
        self,
        x_title: Mobject | None,
        y_title: Mobject | None,
        theme: Theme,
        sides: tuple[str, str] = ("bottom", "left"),
    ) -> None:
        """Place the shared axis titles outside the grid (centred on the plotting
        area) and add them."""
        from manim import PI

        lo = self.axes.get_corner(np.array([-1.0, -1.0, 0.0]))
        hi = self.axes.get_corner(np.array([1.0, 1.0, 0.0]))
        buff = theme.axis_title_buff
        body = VGroup(*self.submobjects)
        if x_title is not None:
            x_title.next_to(body, DOWN if sides[0] == "bottom" else UP, buff=buff)
            x_title.set_x((lo[0] + hi[0]) / 2)
            self.x_title = x_title
        if y_title is not None:
            right = sides[1] == "right"
            y_title.rotate(-PI / 2 if right else PI / 2)
            y_title.next_to(body, RIGHT if right else LEFT, buff=buff)
            y_title.set_y((lo[1] + hi[1]) / 2)
            self.y_title = y_title
        for m in (self.x_title, self.y_title):
            if m is not None:
                self.add(m)  # type: ignore[arg-type]


class PanelLayers(VGroup):
    """One layer of a faceted plot: a ``VGroup`` per panel (the geom's output for
    that panel's rows), in ``PANEL`` order.

    ``.panel(row, col)`` gives one panel's group; ``.marks`` / ``.keys`` are the
    per-panel marks/keys concatenated in panel order; ``.parts`` is
    ``{(row, col): VGroup}``.
    """

    def __init__(self, parts: Mapping[tuple[int, int], Any], **kwargs):
        super().__init__(**kwargs)
        self.parts = dict(parts)
        self.add(*self.parts.values())
        self.marks: list = []
        self.keys: list = []
        for part in self.parts.values():
            self.marks.extend(getattr(part, "marks", None) or [])
            self.keys.extend(getattr(part, "keys", None) or [])

    def panel(self, row: int, col: int) -> Mobject:
        """This layer's mobject in panel ``(row, col)``."""
        return self.parts[(row, col)]
