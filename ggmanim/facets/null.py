"""``facet_null()``: a single panel (the default when a plot has no facet)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import pandas as pd

from ..registry import register
from .base import Facet

__all__ = ["FacetNull", "facet_null"]


class FacetNull(Facet):
    """No faceting: one panel holding every row. The pipeline builds the classic
    single-panel plot (``GGPlot.panel``, axis titles on the panel, no strips)."""

    is_null: ClassVar[bool] = True

    def __init__(self):
        super().__init__(scales="fixed")

    def __repr__(self):
        return "facet_null()"

    def compute_layout(self, data: Sequence[pd.DataFrame]) -> pd.DataFrame:
        return pd.DataFrame({"PANEL": [1], "ROW": [1], "COL": [1], "SCALE_X": [1], "SCALE_Y": [1]})

    def map_data(self, data: pd.DataFrame, layout: pd.DataFrame) -> pd.DataFrame:
        """``PANEL = 1`` on non-empty data (empty layers get it after the stat)."""
        return data.assign(PANEL=1) if len(data) else data


@register
def facet_null() -> FacetNull:
    """No facets: a single panel (the default)."""
    return FacetNull()
