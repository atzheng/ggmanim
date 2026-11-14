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


def _num(data: pd.DataFrame, col: str) -> np.ndarray:
    return np.asarray(pd.to_numeric(data[col], errors="coerce"), dtype=float)


def _segmented_cumsum(y: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """``np.cumsum`` restarted at every ``True`` in ``starts``, with the same
    left-to-right float additions as one ``np.cumsum`` per segment (pandas'
    grouped cumsum uses compensated summation, which rounds differently)."""
    n = len(y)
    first = np.flatnonzero(starts)
    if n == 0:
        return y.copy()
    lengths = np.diff(np.r_[first, n])
    if lengths.max() <= len(first):
        # few rows per stack (e.g. one bar per x): one vectorised step per rank
        top = y.copy()
        rank = np.arange(n) - np.repeat(first, lengths)
        for r in range(1, int(lengths.max())):
            idx = np.flatnonzero(rank == r)
            top[idx] = top[idx - 1] + y[idx]
        return top
    # few, tall stacks: one cumsum per stack
    return np.concatenate([np.cumsum(seg) for seg in np.split(y, first[1:])])


def _stack_frame(data: pd.DataFrame, vjust: float, fill: bool, reverse: bool) -> pd.DataFrame:
    """Stack all rows of one panel at once (vectorised).

    Stacks are formed per (sign of ``ymax``, x); within a stack rows are taken in
    descending ``group`` order (ascending with ``reverse``), stably, and their
    ``y`` (NaN as 0) accumulated from 0. Rows keep their order in ``data``.
    """
    n = len(data)
    ymax = _num(data, "ymax")
    negative = np.nan_to_num(ymax, nan=0.0) < 0
    xcol = "xmin" if "xmin" in data.columns else "x"
    key = _num(data, xcol)
    group = _num(data, "group") if "group" in data.columns else np.zeros(n)
    g = group if reverse else -group
    # stacking order: sign, then x (NaN last), then group (NaN last); stable
    order = np.lexsort((g, key, negative))
    ks = key[order]
    neg_s = negative[order]
    # a new stack starts where the sign or x changes (all NaN x share one stack)
    same_key = (ks[1:] == ks[:-1]) | (np.isnan(ks[1:]) & np.isnan(ks[:-1]))
    starts = np.ones(n, dtype=bool)
    starts[1:] = ~(same_key & (neg_s[1:] == neg_s[:-1]))
    stack_id = np.cumsum(starts) - 1

    y = _num(data, "y")[order]
    y = np.where(np.isnan(y), 0.0, y)
    top = _segmented_cumsum(y, starts)
    bottom = np.empty(n)
    bottom[1:] = top[:-1]
    bottom[starts] = 0.0
    if fill:
        last = np.r_[starts[1:], True]
        total = np.abs(top[last])[stack_id]
        ok = total > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            top = np.where(ok, top / np.where(ok, total, 1.0), np.nan)
            bottom = np.where(ok, bottom / np.where(ok, total, 1.0), np.nan)
    if "ymin" in data.columns and "ymax" in data.columns:
        max_is_lower = (ymax < _num(data, "ymin"))[order]
    else:
        max_is_lower = np.zeros(n, dtype=bool)
    lo = np.minimum(bottom, top)
    hi = np.maximum(bottom, top)

    inv = np.empty(n, dtype=np.intp)
    inv[order] = np.arange(n)
    out = data.copy()
    out["y"] = ((1.0 - vjust) * lo + vjust * hi)[inv]
    out["ymin"] = np.where(max_is_lower, hi, lo)[inv]
    out["ymax"] = np.where(max_is_lower, lo, hi)[inv]
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
        out = _stack_frame(
            data.reset_index(drop=True), params["vjust"], params["fill"], params["reverse"]
        )
        out.index = index
        return out


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
