"""``position_dodge`` and ``position_dodge2``: place overlapping objects side by
side (ggplot2 semantics).

``position_dodge`` divides the width of each x position (the bars' ``xmax - xmin``,
or ``width=``) among the groups found at that x. ``position_dodge2`` works on
intervals instead: rows whose ``[xmin, xmax]`` overlap are placed next to each
other, each keeping a share of the overlap proportional to its own width, with
``padding`` between them; it handles variable widths (e.g. boxplots with
``varwidth``) and needs no grouping.

``preserve="total"`` (default) lets all the rows at one x share its full width;
``preserve="single"`` gives every row the width of one slot of the busiest x in
the panel, so bars have equal widths everywhere.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from .base import Position

__all__ = ["PositionDodge", "PositionDodge2", "position_dodge", "position_dodge2"]


def _num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _check_preserve(preserve: str) -> str:
    if preserve not in ("total", "single"):
        raise ValueError(f"preserve must be 'total' or 'single', got {preserve!r}")
    return preserve


def collide_setup(
    data: pd.DataFrame, width: float | None, reverse: bool = False
) -> tuple[pd.DataFrame, float | None]:
    """ggplot2's ``collide_setup``: make sure ``xmin``/``xmax`` exist (``x -/+ width/2``,
    or ``x`` itself without a width), resolve ``width`` from the data when ``None``,
    and order the rows by ``(xmin, group)`` (group descending when ``reverse``).
    ``data`` must have a unique index."""
    data = data.copy()
    has_ext = "xmin" in data.columns and "xmax" in data.columns
    if width is not None:
        if not has_ext:
            x = _num(data["x"])
            data["xmin"] = x - width / 2
            data["xmax"] = x + width / 2
    else:
        if not has_ext:
            data["xmin"] = data["x"]
            data["xmax"] = data["x"]
        widths = _num(data["xmax"]) - _num(data["xmin"])
        widths = widths[np.isfinite(widths)]
        width = float(widths[0]) if len(widths) else None
    group = _num(data["group"]) if "group" in data.columns else np.zeros(len(data))
    key = pd.DataFrame({"k": _num(data["xmin"]), "g": -group if reverse else group}, data.index)
    return data.loc[key.sort_values(["k", "g"], kind="stable").index], width


def pos_dodge(df: pd.DataFrame, width: float | None, n: int | None = None) -> pd.DataFrame:
    """Dodge the rows of one x position (ggplot2's ``pos_dodge``)."""
    groups = pd.unique(df["group"]) if "group" in df.columns else np.array([0])
    if n is None:
        n = len(groups)
    if n == 1:
        return df
    df = df.copy()
    xmin, xmax = _num(df["xmin"]), _num(df["xmax"])
    d_width = float(np.nanmax(xmax - xmin)) if len(df) else 0.0
    w = d_width if width is None else width
    gid = pd.Index(groups).get_indexer(df["group"]) + 1 if "group" in df.columns else 1
    x = _num(df["x"]) if "x" in df.columns else (xmin + xmax) / 2
    x = x + w * ((gid - 0.5) / n - 0.5)
    df["x"] = x
    df["xmin"] = x - d_width / n / 2
    df["xmax"] = x + d_width / n / 2
    return df


def _restore(pieces: list[pd.DataFrame], index: pd.Index) -> pd.DataFrame:
    out = pd.concat(pieces).sort_index()
    out.index = index
    return out


class PositionDodge(Position):
    """Dodge overlapping objects side to side (see the module docstring)."""

    name = "dodge"

    def __init__(
        self, width: float | None = None, preserve: str = "total", reverse: bool = False
    ) -> None:
        super().__init__(width=width, preserve=_check_preserve(preserve), reverse=reverse)

    def setup_params(self, data: pd.DataFrame) -> dict[str, Any]:
        width = self.params.get("width")
        if width is None and "xmin" not in data.columns and "xmax" not in data.columns:
            warnings.warn("Width not defined. Set with `position_dodge(width = ...)`", stacklevel=2)
        n = None
        if self.params.get("preserve") == "single" and len(data):
            n = _max_groups_per_x(data)
        return {"width": width, "n": n, "reverse": bool(self.params.get("reverse", False))}

    def setup_data(self, data: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        data = super().setup_data(data, params)
        if "x" not in data.columns and {"xmin", "xmax"} <= set(data.columns):
            data = data.assign(x=(_num(data["xmin"]) + _num(data["xmax"])) / 2)
        return data

    def compute_panel(
        self, data: pd.DataFrame, params: dict[str, Any], scales: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        index = data.index
        data, width = collide_setup(data.reset_index(drop=True), params["width"], params["reverse"])
        key = _num(data["xmin"])
        pieces = [
            pos_dodge(part, width, params["n"])
            for _, part in data.groupby(key, sort=False, dropna=False)
        ]
        return _restore(pieces, index)


def _max_groups_per_x(data: pd.DataFrame) -> int:
    """Most distinct groups at one (PANEL, x) -- ``preserve="single"`` for dodge."""
    xcol = "xmin" if "xmin" in data.columns else "x"
    cols = [c for c in ("PANEL",) if c in data.columns]
    keys = data[cols].assign(_x=_num(data[xcol]))
    group = data["group"] if "group" in data.columns else pd.Series(0, index=data.index)
    counts = group.groupby([keys[c] for c in keys.columns], dropna=False).nunique()
    return int(counts.max()) if len(counts) else 1


def _max_rows_per_x(data: pd.DataFrame) -> int:
    """Most rows at one (PANEL, x) -- ``preserve="single"`` for dodge2."""
    xcol = "xmin" if "xmin" in data.columns else "x"
    keys = [data[c] for c in ("PANEL",) if c in data.columns] + [
        pd.Series(_num(data[xcol]), index=data.index)
    ]
    counts = data.groupby(keys, dropna=False).size()
    return int(counts.max()) if len(counts) else 1


def _find_x_overlaps(xmin: np.ndarray, xmax: np.ndarray) -> np.ndarray:
    """1-based id of the run of overlapping intervals each (xmin-sorted) row is in.
    Intervals that merely touch do not overlap."""
    n = len(xmin)
    ids = np.ones(n, dtype=int)
    if n == 0:
        return ids
    lo = np.where(np.isnan(xmin), xmax, xmin)
    hi = np.where(np.isnan(xmax), xmin, xmax)
    end = hi[0]
    cur = 1
    for i in range(1, n):
        if lo[i] >= end - 1e-9 * max(1.0, abs(end)):
            cur += 1
            end = hi[i]
        else:
            end = max(end, hi[i])
        ids[i] = cur
    return ids


def pos_dodge2(df: pd.DataFrame, n: int | None = None, padding: float = 0.1) -> pd.DataFrame:
    """Dodge xmin-sorted rows by interval overlap (ggplot2's ``pos_dodge2``)."""
    df = df.copy()
    if not {"xmin", "xmax"} <= set(df.columns):
        df["xmin"] = df["x"]
        df["xmax"] = df["x"]
    xmin, xmax = _num(df["xmin"]), _num(df["xmax"])
    xid = _find_x_overlaps(xmin, xmax)
    counts = np.bincount(xid)[xid]
    new_width = (xmax - xmin) / (counts if n is None else n)
    new_xmin = np.empty_like(xmin)
    new_xmax = np.empty_like(xmax)
    for i in np.unique(xid):
        m = xid == i
        start = np.nanmin(xmin[m])
        if n is not None:  # preserve="single": centre the slots in the original span
            span = np.nanmax(xmax[m]) - start
            start += (span - np.nansum(new_width[m])) / 2
        div = start + np.concatenate([[0.0], np.cumsum(new_width[m])])
        new_xmin[m] = div[:-1]
        new_xmax[m] = div[1:]
    x = (new_xmin + new_xmax) / 2
    df["x"] = x
    if not np.any(np.bincount(xid)[1:] > 1):  # nothing shares a position: no padding
        df["xmin"], df["xmax"] = new_xmin, new_xmax
        return df
    pad = new_width * (1 - padding)
    df["xmin"] = x - pad / 2
    df["xmax"] = x + pad / 2
    return df


class PositionDodge2(Position):
    """Dodge overlapping intervals side to side, with padding (see the module
    docstring)."""

    name = "dodge2"

    def __init__(
        self,
        width: float | None = None,
        preserve: str = "total",
        padding: float = 0.1,
        reverse: bool = False,
    ) -> None:
        super().__init__(
            width=width, preserve=_check_preserve(preserve), padding=padding, reverse=reverse
        )

    def setup_params(self, data: pd.DataFrame) -> dict[str, Any]:
        n = None
        if self.params.get("preserve") == "single" and len(data):
            n = _max_rows_per_x(data)
        return {
            "width": self.params.get("width"),
            "n": n,
            "padding": float(self.params.get("padding", 0.1)),
            "reverse": bool(self.params.get("reverse", False)),
        }

    def setup_data(self, data: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        data = super().setup_data(data, params)
        if "x" not in data.columns and {"xmin", "xmax"} <= set(data.columns):
            data = data.assign(x=(_num(data["xmin"]) + _num(data["xmax"])) / 2)
        return data

    def compute_panel(
        self, data: pd.DataFrame, params: dict[str, Any], scales: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        index = data.index
        data, _ = collide_setup(data.reset_index(drop=True), params["width"], params["reverse"])
        out = pos_dodge2(data, params["n"], params["padding"])
        return _restore([out], index)


def position_dodge(
    width: float | None = None, preserve: str = "total", reverse: bool = False
) -> PositionDodge:
    """Dodge overlapping objects side to side (ggplot2's ``position_dodge``).

    Parameters
    ----------
    width : float, optional
        Dodging width in x (transformed) units; defaults to the width of the
        objects (``xmax - xmin``, e.g. ``0.9`` for bars). Needed for points.
    preserve : {"total", "single"}
        Share the width of an x among its groups, or keep every bar the width of
        one slot of the busiest x.
    reverse : bool
        Dodge the groups in reverse order.
    """
    return PositionDodge(width=width, preserve=preserve, reverse=reverse)


def position_dodge2(
    width: float | None = None,
    preserve: str = "total",
    padding: float = 0.1,
    reverse: bool = False,
) -> PositionDodge2:
    """Dodge by interval overlap, for variable widths (ggplot2's ``position_dodge2``).

    ``padding`` is the fraction of each slot left empty between neighbours
    (``0.1``). The default for boxplots.
    """
    return PositionDodge2(width=width, preserve=preserve, padding=padding, reverse=reverse)
