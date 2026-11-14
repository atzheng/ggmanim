"""stat_sum: counts of observations at each distinct position (for geom_count)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..aes import after_stat
from ..registry import register
from .base import Stat

__all__ = ["StatSum", "stat_sum"]


class StatSum(Stat):
    """Counts rows (or sums ``weight``) at each distinct ``(x, y)`` per group.

    Output columns: ``x``, ``y``, ``n`` (count / summed weight) and ``prop``
    (``n`` over the group's total), in order of first appearance. Other columns
    constant within the group are carried over. Map ``size=after_stat("n")`` (as
    :func:`~ggmanim.geoms.count.geom_count` does) or ``after_stat("prop")``.
    """

    required_aes = ("x", "y")
    #: Applied by the layer constructors when ``size`` is neither mapped nor set.
    default_aes = {"size": after_stat("n")}
    dropped_aes = ("weight",)

    def compute_group(self, df: pd.DataFrame, scales=None) -> pd.DataFrame:  # type: ignore[override]
        w = (
            pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if "weight" in df.columns
            else np.ones(len(df))
        )
        tmp = pd.DataFrame({"x": df["x"].to_numpy(), "y": df["y"].to_numpy(), "n": w})
        out = tmp.groupby(["x", "y"], sort=False, dropna=False, observed=True)["n"].sum()
        out = out.reset_index()
        total = out["n"].sum()
        out["prop"] = out["n"] / total if total else np.nan
        return out


@register
def stat_sum(
    mapping=None,
    data=None,
    *,
    geom="point",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Layer of points sized by the number of observations at each position
    (same as :func:`~ggmanim.geoms.count.geom_count`). Returns a layer."""
    from ..geoms.count import stat_layer  # geoms import manim: keep stats free of it

    return stat_layer(
        "sum",
        geom,
        mapping,
        data,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
