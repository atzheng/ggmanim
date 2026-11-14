"""``position_stack`` and ``position_fill``: stack overlapping objects on top of
each other (ggplot2 semantics).

Rows are stacked per x position (``xmin`` when present, else ``x``) and per
``PANEL``. Within an x, rows are stacked in *reverse* group order, so the first
group ends up on top, as in ggplot2 (``reverse=True`` flips this). Negative values
are stacked separately, downwards from 0. ``position_fill`` additionally divides
each stack by its total so it spans ``[0, 1]`` (``[-1, 0]`` for negatives).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import Position

__all__ = ["PositionStack", "PositionFill", "position_stack", "position_fill"]


def _stack_var(data: pd.DataFrame) -> str | None:
    if "ymax" in data.columns:
        return "ymax"
    if "y" in data.columns:
        return "y"
    return None


def _pos_stack(df: pd.DataFrame, vjust: float, fill: bool) -> pd.DataFrame:
    """Stack the rows of ``df`` (one x, already in stacking order)."""
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    y = np.where(np.isnan(y), 0.0, y)
    heights = np.concatenate([[0.0], np.cumsum(y)])
    if fill:
        total = abs(heights[-1])
        heights = heights / total if total > 0 else np.full_like(heights, np.nan)
    if "ymin" in df.columns and "ymax" in df.columns:
        top = pd.to_numeric(df["ymax"], errors="coerce").to_numpy(float)
        bottom = pd.to_numeric(df["ymin"], errors="coerce").to_numpy(float)
        max_is_lower = top < bottom
    else:
        max_is_lower = np.zeros(len(df), dtype=bool)
    lo = np.minimum(heights[:-1], heights[1:])
    hi = np.maximum(heights[:-1], heights[1:])
    out = df.copy()
    out["y"] = (1.0 - vjust) * lo + vjust * hi
    out["ymin"] = np.where(max_is_lower, hi, lo)
    out["ymax"] = np.where(max_is_lower, lo, hi)
    return out


class PositionStack(Position):
    """Stack overlapping rows (see the module docstring).

    Parameters
    ----------
    vjust : float
        Where in its slice a row's ``y`` ends up: ``1`` (default) the top, ``0.5``
        the middle (for centred labels), ``0`` the bottom.
    reverse : bool
        Stack the first group at the bottom instead of the top.
    """

    name = "stack"
    fill: bool = False

    def __init__(self, vjust: float = 1.0, reverse: bool = False) -> None:
        super().__init__(vjust=vjust, reverse=reverse)

    def setup_params(self, data: pd.DataFrame) -> dict[str, Any]:
        return {
            "var": _stack_var(data),
            "vjust": float(self.params.get("vjust", 1.0)),
            "reverse": bool(self.params.get("reverse", False)),
            "fill": self.fill,
        }

    def setup_data(self, data: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        data = super().setup_data(data, params)
        var = params["var"]
        if var is None or len(data) == 0:
            return data
        data = data.copy()
        if var == "y":
            data["ymax"] = data["y"]
        else:
            ymax = pd.to_numeric(data["ymax"], errors="coerce")
            if "ymin" in data.columns:
                ymin = pd.to_numeric(data["ymin"], errors="coerce")
                data["ymax"] = ymax.where(ymax != 0, ymin)
        return data

    def compute_panel(
        self, data: pd.DataFrame, params: dict[str, Any], scales: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        if params["var"] is None or "y" not in data.columns:
            return data
        index = data.index
        data = data.reset_index(drop=True)
        ymax = pd.to_numeric(data["ymax"], errors="coerce").to_numpy(float)
        negative = np.nan_to_num(ymax, nan=0.0) < 0
        pieces = [self._collide(data[mask], params) for mask in (negative, ~negative) if mask.any()]
        out = pd.concat(pieces).sort_index()
        out.index = index
        return out

    @staticmethod
    def _collide(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        xcol = "xmin" if "xmin" in df.columns else "x"
        key = pd.to_numeric(df[xcol], errors="coerce")
        group = pd.to_numeric(df["group"], errors="coerce") if "group" in df.columns else 0
        order = pd.DataFrame({"k": key, "g": group if params["reverse"] else -group})
        df = df.loc[order.sort_values(["k", "g"], kind="stable").index]
        out = [
            _pos_stack(part, params["vjust"], params["fill"])
            for _, part in df.groupby(key.loc[df.index], sort=False, dropna=False)
        ]
        return pd.concat(out)


class PositionFill(PositionStack):
    """Stack and normalise each stack to height 1 (``position_fill``)."""

    name = "fill"
    fill = True


def position_stack(vjust: float = 1.0, reverse: bool = False) -> PositionStack:
    """Stack overlapping objects on top of each other (ggplot2's ``position_stack``).

    The default for :func:`~ggmanim.geom_col`. ``vjust=0.5`` centres e.g. labels in
    their slice; ``reverse=True`` puts the first group at the bottom.
    """
    return PositionStack(vjust=vjust, reverse=reverse)


def position_fill(vjust: float = 1.0, reverse: bool = False) -> PositionFill:
    """Stack and normalise every stack to ``[0, 1]`` (ggplot2's ``position_fill``)."""
    return PositionFill(vjust=vjust, reverse=reverse)
