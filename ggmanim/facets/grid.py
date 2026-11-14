"""``facet_grid()``: a matrix of panels, rows by one set of variables and
columns by another (ggplot2's ``facet_grid``).

>>> facet_grid(rows="drv", cols="cyl")          # doctest: +SKIP
>>> facet_grid("drv ~ cyl")                     # the formula form
>>> facet_grid(cols=["am", "vs"], scales="free_x", space="free_x")

Layout
------
The grid has one row per observed combination of the ``rows`` variables and one
column per observed combination of the ``cols`` variables (each sorted by the
variables' levels, see :meth:`Facet.combinations
<ggmanim.facets.base.Facet.combinations>`); *every* row x column cell is a panel,
even when no data falls in it. ``drop=False`` uses every level of each variable
(categorical categories) instead of the observed ones.

Scales are shared along the grid: ``scales="free_x"`` gives one x scale per
*column*, ``"free_y"`` one y scale per *row* (so axes are drawn once per
column / row). ``space="free_x"`` / ``"free_y"`` / ``"free"`` then sizes the
columns / rows in proportion to their scale range, so e.g. a discrete x axis
gets the same width per level in every column.

Strips: the column strips go on top of the first row, the row strips on the
right of the last column (text rotated); ``switch="x"`` / ``"y"`` / ``"both"``
moves them to the bottom / left. ``margins=True`` (or a list of facet variables)
adds ``"(all)"`` rows/columns holding every row of the data, as in ggplot2.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ..registry import register
from . import _layout
from .base import Facet, StripSpecs, _isna, _levels

__all__ = ["FacetGrid", "facet_grid", "MARGIN_LABEL"]

#: The facet value of the margin panels (``margins=True``).
MARGIN_LABEL = "(all)"

_SPACES = ("fixed", "free", "free_x", "free_y")
_SWITCH = (None, "x", "y", "both")


def _as_vars(spec: Any, what: str) -> list[str]:
    """``None`` / ``"."`` / ``"a"`` / ``"a + b"`` / ``["a", "b"]`` -> variable names."""
    if spec is None:
        return []
    if isinstance(spec, str):
        parts = [s.strip() for s in spec.split("+")]
        return [s for s in parts if s and s != "."]
    if isinstance(spec, Sequence) and all(isinstance(s, str) for s in spec):
        return [s.strip() for s in spec if s.strip() not in ("", ".")]
    raise TypeError(f"facet_grid {what} must be a name, a list of names or None, got {spec!r}")


def parse_formula(formula: str) -> tuple[list[str], list[str]]:
    """``"a + b ~ c"`` -> ``(["a", "b"], ["c"])``; ``"."`` means no variable."""
    if formula.count("~") != 1:
        raise ValueError(f"facet_grid formula must be 'rows ~ cols', got {formula!r}")
    lhs, rhs = formula.split("~")
    return _as_vars(lhs, "rows"), _as_vars(rhs, "cols")


class FacetGrid(Facet):
    """Panels in a grid: rows by ``rows``, columns by ``cols``. See the module
    docstring and :func:`facet_grid`."""

    def __init__(
        self,
        rows: Any = None,
        cols: Any = None,
        scales: str = "fixed",
        space: str = "fixed",
        margins: bool | Sequence[str] = False,
        labeller: Any = None,
        switch: str | None = None,
        drop: bool = True,
    ):
        super().__init__(scales=scales, labeller=labeller)
        if isinstance(rows, str) and "~" in rows:
            if cols is not None:
                raise ValueError("facet_grid: give a formula 'rows ~ cols' or cols=, not both")
            self.rows, self.cols = parse_formula(rows)
        else:
            self.rows, self.cols = _as_vars(rows, "rows"), _as_vars(cols, "cols")
        both = set(self.rows) & set(self.cols)
        if both:
            raise ValueError(f"facet_grid: {sorted(both)} in both rows and cols")
        if len(set(self.rows)) < len(self.rows) or len(set(self.cols)) < len(self.cols):
            raise ValueError("facet_grid: a variable is repeated")
        if space not in _SPACES:
            raise ValueError(f"space must be one of {_SPACES}, got {space!r}")
        if switch not in _SWITCH:
            raise ValueError(f"switch must be one of {_SWITCH}, got {switch!r}")
        self.space = space
        self.switch = switch
        self.drop = bool(drop)
        if margins is True:
            self.margins = self.vars()
        elif margins is False or margins is None:
            self.margins = []
        else:
            names = [margins] if isinstance(margins, str) else list(margins)
            unknown = [m for m in names if m not in self.vars()]
            if unknown:
                raise ValueError(f"facet_grid margins {unknown} are not facet variables")
            self.margins = names

    def __repr__(self):
        return (
            f"FacetGrid(rows={self.rows!r}, cols={self.cols!r}, scales={self.scales!r}, "
            f"space={self.space!r})"
        )

    def vars(self) -> list[str]:
        return [*self.rows, *self.cols]

    # -- margins ---------------------------------------------------------------
    def margin_sets(self) -> list[set[str]]:
        """The variable sets set to ``"(all)"`` for each margin copy of the data.

        Per dimension (rows, cols) a margin on variable ``v`` sets ``v`` and the
        variables after it (inner ones) to ``"(all)"``; the dimensions combine,
        so ``margins=True`` on ``rows="a", cols="b"`` gives ``{a}``, ``{b}`` and
        ``{a, b}`` (the grand total)."""
        if not self.margins:
            return []
        per_dim = []
        for dim in (self.rows, self.cols):
            options: list[frozenset[str]] = [frozenset()]
            for i, v in enumerate(dim):
                if v in self.margins:
                    options.append(frozenset(dim[i:]))
            per_dim.append(list(dict.fromkeys(options)))
        sets = [set(a | b) for a, b in itertools.product(*per_dim)]
        return [s for s in sets if s]

    def add_margins(self, df: pd.DataFrame) -> pd.DataFrame:
        """``df`` plus one copy per :meth:`margin_sets` entry with those facet
        variables set to ``"(all)"`` (copies that change nothing are skipped)."""
        sets = self.margin_sets()
        if not sets or not len(df):
            return df
        seen: list[frozenset[str]] = []
        pieces = [df]
        for s in sets:
            present = frozenset(v for v in s if v in df.columns)
            if not present or present in seen:
                continue
            seen.append(present)
            copy = df.copy()
            for v in present:
                copy[v] = MARGIN_LABEL
            pieces.append(copy)
        if len(pieces) == 1:
            return df
        for v in {v for s in seen for v in s}:
            pieces = [p.assign(**{v: p[v].astype(object)}) for p in pieces]
        return pd.concat(pieces, ignore_index=True)

    def _ordered_frames(self, data: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
        """Frames with margins added and each margin variable categorical, its
        levels followed by ``"(all)"`` (so margins sort last)."""
        if not self.margins:
            return list(data)
        cats = {}
        for v in {v for s in self.margin_sets() for v in s}:
            levels = _levels([df[v] for df in data if v in df.columns])
            cats[v] = [x for x in levels if not _isna(x)] + [MARGIN_LABEL]
        out = []
        for df in data:
            df = self.add_margins(df)
            for v, c in cats.items():
                if v in df.columns:
                    df = df.assign(**{v: pd.Categorical(df[v].astype(object), categories=c)})
            out.append(df)
        return out

    # -- 1. layout -------------------------------------------------------------
    def _dimension(self, data: Sequence[pd.DataFrame], vars: list[str]) -> pd.DataFrame:
        if not vars or self.drop:
            return self.combinations(data, vars)
        self.combinations(data, vars)  # validates the variables exist
        levels = [_levels([df[v] for df in data if v in df.columns]) for v in vars]
        return pd.MultiIndex.from_product(levels, names=vars).to_frame(index=False)

    def compute_layout(self, data: Sequence[pd.DataFrame]) -> pd.DataFrame:
        frames = self._ordered_frames(data)
        rows = self._dimension(frames, self.rows).reset_index(drop=True)
        cols = self._dimension(frames, self.cols).reset_index(drop=True)
        rows = rows.assign(ROW=range(1, len(rows) + 1))
        cols = cols.assign(COL=range(1, len(cols) + 1))
        layout = rows.merge(cols, how="cross")
        layout = layout.sort_values(["ROW", "COL"], kind="stable").reset_index(drop=True)
        layout.insert(0, "PANEL", range(1, len(layout) + 1))
        for v in self.vars():
            layout[v] = layout[v].astype(object)
        layout = self.scale_ids(layout, x_by="COL", y_by="ROW")
        front = ["PANEL", "ROW", "COL", "SCALE_X", "SCALE_Y"]
        return layout[front + self.vars()]

    # -- 2. rows -> panels -----------------------------------------------------
    def map_data(self, data: pd.DataFrame, layout: pd.DataFrame) -> pd.DataFrame:
        """The generic mapping, after adding the margin copies of the rows."""
        return super().map_data(self.add_margins(data), layout)

    # -- 4. panels -------------------------------------------------------------
    def strips(self, layout: pd.DataFrame) -> StripSpecs:
        """Column strips on the top row (bottom row with ``switch="x"``/``"both"``),
        row strips on the last column (first with ``switch="y"``/``"both"``)."""
        nrow, ncol = int(layout["ROW"].max()), int(layout["COL"].max())
        sw_x = self.switch in ("x", "both")
        sw_y = self.switch in ("y", "both")
        out: StripSpecs = {}
        for r in layout.itertuples(index=False):
            p = int(r.PANEL)
            entries = []
            if self.cols and int(r.ROW) == (nrow if sw_x else 1):
                side = "bottom" if sw_x else "top"
                labels = self.strip_labels(layout, p, self.cols)
                # labels are outermost first: for the top strip that is the
                # first variable; below the panel too (ggplot2 stacks the same)
                entries.append((side, labels))
            if self.rows and int(r.COL) == (1 if sw_y else ncol):
                side = "left" if sw_y else "right"
                entries.append((side, self.strip_labels(layout, p, self.rows)))
            if entries:
                out[p] = entries
        return out

    def panel_weights(
        self, layout: pd.DataFrame, panel_params: Mapping[int, Any]
    ) -> tuple[list[float] | None, list[float] | None]:
        """With ``space="free_x"``/``"free"``: column widths proportional to the
        length of each column's x range; ``"free_y"``/``"free"``: row heights
        proportional to each row's y range.

        Widths/heights follow the *drawn* axes: under ``coord_flip``
        (``PanelParams.flipped``) the columns are sized by the y aesthetic's range
        (drawn horizontally) and the rows by the x aesthetic's."""
        nrow, ncol = int(layout["ROW"].max()), int(layout["COL"].max())
        col_w: list[float] | None = None
        row_w: list[float] | None = None

        def ranges(p: int) -> tuple[Any, Any]:  # (horizontal, vertical) axis ranges
            return _layout.space_ranges(panel_params[p])

        if self.space in ("free", "free_x"):
            col_w = [1.0] * ncol
            for r in layout.itertuples(index=False):
                lo, hi = ranges(int(r.PANEL))[0]
                col_w[int(r.COL) - 1] = _length(lo, hi)
        if self.space in ("free", "free_y"):
            row_w = [1.0] * nrow
            for r in layout.itertuples(index=False):
                lo, hi = ranges(int(r.PANEL))[1]
                row_w[int(r.ROW) - 1] = _length(lo, hi)
        return col_w, row_w


def _length(lo: float, hi: float) -> float:
    span = abs(float(hi) - float(lo))
    return span if span > 0 else 1.0


@register
def facet_grid(
    rows: Any = None,
    cols: Any = None,
    scales: str = "fixed",
    space: str = "fixed",
    margins: bool | Sequence[str] = False,
    labeller: Any = None,
    switch: str | None = None,
    drop: bool = True,
) -> FacetGrid:
    """Lay panels out in a grid: one row per level of ``rows``, one column per
    level of ``cols``.

    Parameters
    ----------
    rows, cols : a variable name, a list of names, ``"a + b"`` or ``None``/``"."``
        (no variable). ``rows`` may also be a formula ``"rows ~ cols"``.
    scales : ``"fixed"``, ``"free_x"`` (one x scale per column), ``"free_y"`` (one
        y scale per row) or ``"free"``.
    space : ``"fixed"`` (equal panels) or ``"free_x"``/``"free_y"``/``"free"``:
        column widths / row heights proportional to their scale ranges (use with
        free scales).
    margins : ``True`` adds ``"(all)"`` panels over every facet variable; a list
        of variable names adds them for those only.
    labeller : strip labeller (:mod:`ggmanim.facets.labellers`).
    switch : ``"x"`` moves the column strips below the grid, ``"y"`` the row
        strips to its left, ``"both"`` both.
    drop : ``False`` keeps unobserved levels (categorical categories).
    """
    return FacetGrid(rows, cols, scales, space, margins, labeller, switch, drop)
