"""stat_unique: removes duplicated rows."""

from __future__ import annotations

import pandas as pd

from ..registry import register
from .base import Stat

__all__ = ["StatUnique", "stat_unique"]


def _hashable(v):
    return v if isinstance(v, (str, int, float, bool, type(None))) else repr(v)


class StatUnique(Stat):
    """Drops rows that duplicate an earlier row in every column (per panel, so the
    ``group`` and ``PANEL`` columns take part). Unhashable values compare by
    ``repr``."""

    def compute_panel(self, df: pd.DataFrame, scales=None, **params) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        try:
            dup = df.duplicated(keep="first")
        except TypeError:  # unhashable cells (lists, colours ...)
            dup = df.map(_hashable).duplicated(keep="first")
        return df[~dup.to_numpy()].reset_index(drop=True)

    def compute_group(self, df: pd.DataFrame, scales=None) -> pd.DataFrame:  # type: ignore[override]
        return self.compute_panel(df, scales)


@register
def stat_unique(
    mapping=None,
    data=None,
    *,
    geom="point",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Layer drawing only the distinct rows of its data (default geom: points).
    Also usable as ``geom_*(stat="unique")``. Returns a layer."""
    from ..geoms.count import stat_layer  # geoms import manim: keep stats free of it

    return stat_layer(
        "unique",
        geom,
        mapping,
        data,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
