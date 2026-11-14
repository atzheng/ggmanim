"""stat_count: number of observations (or summed ``weight``) at each x.

Feeds :func:`~ggmanim.geoms.col.geom_bar`. On a discrete x scale the levels are
already ``1..n`` when the stat runs, so string columns are counted per level.

Orientation (shared with :mod:`ggmanim.stats.bin`): the stat takes an ``x`` *or*
a ``y`` aesthetic, not both. With only ``y`` mapped (or ``orientation="y"``)
it counts per ``y`` instead: the data is computed with x and y swapped and
comes back with the roles swapped and ``flipped_aes = True``, as in ggplot2
(``has_flipped_aes(main_is_orthogonal = FALSE)``). The counted axis is not an
output column: the default ``x``/``y = after_stat("count")`` aesthetics (added
by the layer for whichever of the two is not mapped) supply it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..aes import after_stat
from ..registry import register
from .base import Stat
from .boxplot import flip_names, resolve_flipped

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


def check_one_axis(stat: Stat, df: pd.DataFrame) -> None:
    """ggplot2: ``stat_count()``/``stat_bin()`` take an x *or* a y aesthetic."""
    name = "stat_" + type(stat).__name__[4:].lower()
    has_x, has_y = "x" in df.columns, "y" in df.columns
    if has_x and has_y:
        raise ValueError(
            f"{name}() must only have an x or y aesthetic (the other is computed: "
            "after_stat('count')); use geom_col() for bars with heights from the data"
        )
    if not (has_x or has_y):
        raise ValueError(f"{name}() requires an x or y aesthetic")


def orient(
    stat: Stat, df: pd.DataFrame, scales=None, orientation: str | None = None
) -> tuple[bool, pd.DataFrame, dict | None]:
    """``(flipped, df, scales)`` with x and y swapped (columns and scales) when the
    layer is horizontal: ``orientation="y"``, or only ``y`` mapped."""
    check_one_axis(stat, df)
    flipped = resolve_flipped(df, scales, orientation, main_is_orthogonal=False)
    if not flipped:
        return False, df, scales
    swapped = {"x": scales.get("y"), "y": scales.get("x")} if scales else scales
    return True, flip_names(df), swapped


def unflip(out: pd.DataFrame, flipped: bool) -> pd.DataFrame:
    """``out`` (computed on swapped columns) with x and y swapped back."""
    return flip_names(out).assign(flipped_aes=True) if flipped else out


class StatCount(Stat):
    """Counts rows (or sums ``weight``) at each distinct ``x`` per group.

    Output columns (one row per distinct x, sorted): ``x``, ``count``, ``prop``
    (``count`` over the group's total), ``width`` and ``flipped_aes = False``.
    ``y`` comes from the default ``y=after_stat("count")`` mapping. ``width``
    defaults to ``0.9 * resolution(x)`` over the whole panel (``0.9`` on a
    discrete scale). Map ``y=after_stat("prop")`` (with ``group=1``) for
    proportions.

    With only ``y`` mapped, or ``orientation="y"``, the roles swap: one row per
    distinct ``y`` and ``flipped_aes = True`` (``x=after_stat("count")``).
    """

    #: Either one (see :func:`check_one_axis`); both are listed so the build drops
    #: rows whose x or y is missing before the stat.
    required_aes = ("x", "y")
    default_aes = {"x": after_stat("count"), "y": after_stat("count")}
    dropped_aes = ("weight",)
    #: The layer runs its position on swapped x/y for horizontal (flipped) data.
    oriented_position = True

    def check_required_aes(self, df: pd.DataFrame) -> None:
        check_one_axis(self, df)

    def compute_panel(  # type: ignore[override]
        self, df: pd.DataFrame, scales=None, width=None, orientation=None, **params
    ) -> pd.DataFrame:
        flipped, df, scales = orient(self, df, scales, orientation)
        if width is None and len(df):
            width = 0.9 * _resolution(df["x"])
        out = super().compute_panel(df, scales, width=width, **params)
        return unflip(out, flipped)

    def compute_group(  # type: ignore[override]
        self, df: pd.DataFrame, scales=None, width=None, orientation=None
    ) -> pd.DataFrame:
        x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
        tmp = pd.DataFrame({"x": x, "count": weights(df)})
        out = tmp.groupby("x", sort=True)["count"].sum().reset_index()
        total = float(np.abs(out["count"]).sum())
        out["prop"] = out["count"] / total if total else np.nan
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
    aesthetic sums weights instead of counting. Map ``y`` instead of ``x`` (or
    pass ``orientation="y"``) for horizontal bars. Returns a layer."""
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
