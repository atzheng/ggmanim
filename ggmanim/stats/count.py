"""stat_count: number of observations (or summed ``weight``) at each x.

Feeds :func:`~ggmanim.geoms.col.geom_bar`. On a discrete x scale the levels are
already ``1..n`` when the stat runs, so string columns are counted per level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..aes import after_stat
from ..registry import register
from .base import Stat

__all__ = ["StatCount", "stat_count"]


def _resolution(x) -> float:
    """Smallest gap between distinct finite values of ``x`` (1 if fewer than two).
    Same as :func:`ggmanim.geoms.col.resolution` (kept here: stats never import manim)."""
    x = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce"), dtype=float)
    u = np.unique(x[np.isfinite(x)])
    if len(u) < 2:
        return 1.0
    d = np.diff(u)
    d = d[d > 1e-9 * max(1.0, float(np.abs(u).max()))]
    return float(d.min()) if len(d) else 1.0


def weights(df: pd.DataFrame) -> np.ndarray:
    """The ``weight`` column as floats (missing -> 0), or ones when absent."""
    if "weight" not in df.columns:
        return np.ones(len(df))
    return pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def check_x_only(stat: Stat, df: pd.DataFrame) -> None:
    """ggplot2: ``stat_count()``/``stat_bin()`` must only have an x aesthetic."""
    if "y" in df.columns:
        name = "stat_" + type(stat).__name__[4:].lower()
        raise ValueError(
            f"{name}() must only have an x aesthetic (y is computed: after_stat('count')); "
            "use geom_col() for bars with heights from the data"
        )


class StatCount(Stat):
    """Counts rows (or sums ``weight``) at each distinct ``x`` per group.

    Output columns (one row per distinct x, sorted): ``x``, ``count``, ``prop``
    (``count`` over the group's total), ``width`` and ``flipped_aes = False``;
    plus ``y = count`` so a geom built without the stat's default mapping (e.g.
    ``geom_text(stat="count")``) still has a y. ``width`` defaults to
    ``0.9 * resolution(x)`` over the whole panel (``0.9`` on a discrete scale).
    Map ``y=after_stat("prop")`` (with ``group=1``) for proportions.
    """

    required_aes = ("x",)
    default_aes = {"y": after_stat("count")}
    dropped_aes = ("weight",)

    def compute_panel(self, df: pd.DataFrame, scales=None, width=None, **params) -> pd.DataFrame:
        check_x_only(self, df)
        if width is None and len(df):
            width = 0.9 * _resolution(df["x"])
        return super().compute_panel(df, scales, width=width, **params)

    def compute_group(self, df: pd.DataFrame, scales=None, width=None) -> pd.DataFrame:  # type: ignore[override]
        x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
        tmp = pd.DataFrame({"x": x, "count": weights(df)})
        out = tmp.groupby("x", sort=True)["count"].sum().reset_index()
        total = float(np.abs(out["count"]).sum())
        out["prop"] = out["count"] / total if total else np.nan
        out["y"] = out["count"]
        out["width"] = 0.9 * _resolution(x) if width is None else width
        out["flipped_aes"] = False
        return out


@register
def stat_count(
    mapping=None,
    data=None,
    *,
    geom="bar",
    position="stack",
    width=None,
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Bars of the number of rows at each x (as :func:`~ggmanim.geoms.col.geom_bar`).
    ``y=after_stat("count")`` by default; ``after_stat("prop")`` too. A ``weight``
    aesthetic sums weights instead of counting. Returns a layer."""
    from ..geoms.count import stat_layer  # geoms import manim: keep stats free of it

    if width is not None:
        params["width"] = width
    return stat_layer(
        "count",
        geom,
        mapping,
        data,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
