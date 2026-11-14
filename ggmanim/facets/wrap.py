"""``facet_wrap()``: a 1-d ribbon of panels wrapped into a 2-d grid.

One panel per observed combination of the facet variables, filled row by row
(``dir="h"``) or column by column (``dir="v"``) into ``nrow`` x ``ncol`` cells.
The layout follows ggplot2's ``FacetWrap``:

* **Grid shape.** With neither ``nrow`` nor ``ncol`` given, ``ncol =
  ceil(sqrt(n))`` and ``nrow = ceil(n / ncol)`` (ggplot2's ``n2mfrow`` rule; the
  one difference is ``n == 3``, which ggplot2 also lays out as 1 x 3). Given one,
  the other is ``ceil(n / given)``. ``dir="v"`` fills columns first (the roles of
  ``nrow``/``ncol`` in the default rule swap, as in ggplot2). ``as_table=False``
  flips the rows, so the first panel is bottom-left.
* **Scales.** ``SCALE_X``/``SCALE_Y`` are the panel id for a free scale, else 1.
* **Axes.** The base rule (:func:`ggmanim.facets._layout.axis_visibility`): with
  fixed scales x tick labels go on the bottom panel of each column (so a panel
  above a gap in an incomplete last row carries them), y on the leftmost panel of
  each row; free scales draw them on every panel.
* **Strips.** One per panel, on ``strip_position`` (``"top"``, ``"bottom"``,
  ``"left"``, ``"right"``), stacked one line per variable with the labeller's
  ``multi_line`` behaviour.

Computed facets
---------------
``facets`` may be a dict ``{name: fn}`` (or a bare callable, named after the
function) where ``fn(df)`` returns the facet values for a data frame, like
``vars(big = cyl > 4)`` in ggplot2. The columns are computed at build time
(:meth:`FacetWrap.facet_data`) on the plot data and on every layer's data, so
layers added after the facet are split too; layers whose data lack the columns
``fn`` needs repeat in every panel. The user's data frames are not modified, and
computed columns are not available to aesthetics.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from ..registry import register
from .base import Facet, _levels

__all__ = ["FacetWrap", "facet_wrap", "wrap_dims"]

_DIRS = ("h", "v")
_SIDES = ("top", "bottom", "left", "right")


def wrap_dims(n: int, nrow: int | None = None, ncol: int | None = None) -> tuple[int, int]:
    """``(nrow, ncol)`` of a wrap of ``n`` panels (ggplot2's ``wrap_dims``)."""
    if n < 1:
        return 1, 1
    if nrow is None and ncol is None:
        ncol = n if n <= 3 else math.ceil(math.sqrt(n))
        nrow = math.ceil(n / ncol)
    elif ncol is None:
        assert nrow is not None
        ncol = math.ceil(n / nrow)
    elif nrow is None:
        nrow = math.ceil(n / ncol)
    if nrow * ncol < n:
        raise ValueError(f"nrow * ncol ({nrow} * {ncol}) must be at least the {n} panels")
    return int(nrow), int(ncol)


def _parse_facets(facets: Any) -> tuple[list[str], dict[str, Callable]]:
    """``(variable names, {name: fn} of computed ones)``."""
    if isinstance(facets, str):
        text = facets.strip().lstrip("~").strip()
        names = [s.strip() for s in text.split("+")] if text else []
        if not names or any(not s for s in names):
            raise ValueError(f"cannot parse facets {facets!r}")
        return names, {}
    if isinstance(facets, Mapping):
        names, computed = [], {}
        for k, v in facets.items():
            names.append(str(k))
            if callable(v):
                computed[str(k)] = v
            elif isinstance(v, str):
                computed[str(k)] = _column(v)
            else:
                raise TypeError(f"facet {k!r} must be a function of the data or a column name")
        return names, computed
    if callable(facets):
        name = getattr(facets, "__name__", "facet")
        name = "facet" if name == "<lambda>" else name
        return [name], {name: facets}
    if isinstance(facets, Sequence):
        names = []
        for f in facets:
            if not isinstance(f, str):
                raise TypeError(f"facets must be column names, got {f!r}")
            names.append(f)
        if not names:
            raise ValueError("facet_wrap needs at least one facet variable")
        return names, {}
    raise TypeError(f"facets must be a str, list of str, dict or callable, got {facets!r}")


def _column(name: str) -> Callable:
    def get(df: pd.DataFrame):
        return df[name]

    return get


class FacetWrap(Facet):
    """``facet_wrap``: see the module docstring and :func:`facet_wrap`."""

    def __init__(
        self,
        facets: Any,
        nrow: int | None = None,
        ncol: int | None = None,
        scales: str = "fixed",
        labeller: Any = None,
        dir: str = "h",
        as_table: bool = True,
        drop: bool = True,
        strip_position: str = "top",
    ):
        super().__init__(scales=scales, labeller=labeller)
        self.facets, self.computed = _parse_facets(facets)
        for name, val in (("nrow", nrow), ("ncol", ncol)):
            if val is not None and (int(val) != val or val < 1):
                raise ValueError(f"{name} must be a positive integer, got {val!r}")
        if dir not in _DIRS:
            raise ValueError(f"dir must be one of {_DIRS}, got {dir!r}")
        if strip_position not in _SIDES:
            raise ValueError(f"strip_position must be one of {_SIDES}, got {strip_position!r}")
        self.nrow = None if nrow is None else int(nrow)
        self.ncol = None if ncol is None else int(ncol)
        self.dir = dir
        self.as_table = bool(as_table)
        self.drop = bool(drop)
        self.strip_position = strip_position

    def __repr__(self):
        return f"facet_wrap({self.facets!r}, scales={self.scales!r})"

    # -- computed facets -------------------------------------------------------
    def facet_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """``data`` plus the computed facet columns that can be evaluated on it."""
        if not self.computed:
            return data
        new = {}
        for name, fn in self.computed.items():
            try:
                values = fn(data)
            except (KeyError, AttributeError):
                continue  # this frame lacks what fn needs: repeat it in every panel
            if hasattr(values, "to_numpy"):
                values = values.to_numpy()
            new[name] = values
        return data.assign(**new) if new else data

    # -- layout ----------------------------------------------------------------
    def vars(self) -> list[str]:
        return list(self.facets)

    def compute_layout(self, data: Sequence[pd.DataFrame]) -> pd.DataFrame:
        if self.drop:
            combos = self.combinations(data)
        else:
            self.combinations(data)  # validates the variables
            levels = [_levels([df[v] for df in data if v in df.columns]) for v in self.facets]
            combos = pd.MultiIndex.from_product(levels, names=self.facets).to_frame(index=False)
        n = len(combos)
        nrow, ncol = self.nrow, self.ncol
        if self.dir == "v":
            nrow, ncol = ncol, nrow  # the default rule is computed for the swapped grid
        dims = wrap_dims(n, nrow, ncol)
        if self.dir == "v":
            dims = dims[::-1]
        nrow, ncol = dims
        idx = range(n)
        if self.dir == "h":
            rows = [i // ncol + 1 for i in idx]
            cols = [i % ncol + 1 for i in idx]
        else:
            rows = [i % nrow + 1 for i in idx]
            cols = [i // nrow + 1 for i in idx]
        if not self.as_table:
            used = max(rows) if rows else 1
            rows = [used - r + 1 for r in rows]
        layout = combos.reset_index(drop=True).assign(PANEL=range(1, n + 1), ROW=rows, COL=cols)
        layout = self.scale_ids(layout)
        cols_order = ["PANEL", "ROW", "COL", "SCALE_X", "SCALE_Y", *self.facets]
        return layout[cols_order]

    # -- strips ----------------------------------------------------------------
    def strips(self, layout: pd.DataFrame) -> dict[int, list[tuple[str, list[str]]]]:
        return {
            int(p): [(self.strip_position, self.strip_labels(layout, int(p), self.facets))]
            for p in layout["PANEL"]
        }


@register
def facet_wrap(
    facets: Any,
    nrow: int | None = None,
    ncol: int | None = None,
    scales: str = "fixed",
    labeller: Any = None,
    dir: str = "h",
    as_table: bool = True,
    drop: bool = True,
    strip_position: str = "top",
) -> FacetWrap:
    """Wrap one panel per level (combination) of ``facets`` into a grid.

    Parameters
    ----------
    facets : a column name (``"class"``, also ``"~ a + b"``), a list of column
        names, or computed facets ``{name: fn(df) | column}`` / a callable.
    nrow, ncol : grid size; default ``ncol = ceil(sqrt(n))``, ``nrow = ceil(n/ncol)``.
    scales : ``"fixed"``, ``"free_x"``, ``"free_y"`` or ``"free"``; free scales
        draw axes on every panel.
    labeller : strip labeller (``label_value`` default, ``label_both``,
        ``labeller(var=dict|fn)``, or a dict / function applied to every value).
    dir : ``"h"`` fills rows first, ``"v"`` columns first.
    as_table : ``True`` puts the first panel top-left; ``False`` bottom-left.
    drop : ``False`` keeps every combination of the variables' levels (including
        unused categories of a categorical column) as a panel, even if empty.
    strip_position : ``"top"``, ``"bottom"``, ``"left"`` or ``"right"``.
    """
    return FacetWrap(
        facets,
        nrow=nrow,
        ncol=ncol,
        scales=scales,
        labeller=labeller,
        dir=dir,
        as_table=as_table,
        drop=drop,
        strip_position=strip_position,
    )
