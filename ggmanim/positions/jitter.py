"""``position_jitter`` and ``position_jitterdodge`` (``geom_jitter`` lives in
:mod:`ggmanim.geoms.jitter`).

Jitter adds uniform random noise to the x and y of every row (all x-like columns
get the row's x offset, all y-like columns its y offset, so bar extents and error
bars move with their row). Noise comes from :func:`numpy.random.default_rng`, so a
given ``seed`` gives the same result on every platform. With ``seed=None`` (the
default) the seed is derived from a hash of the layer's positions, so building a
plot twice gives identical output (:func:`~ggmanim.build.build_plot` is pure).

Defaults follow ggplot2: ``width``/``height`` are 40% of the resolution of the
data (on a discrete axis the resolution is 1, so points stay within their level).
"""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np
import pandas as pd

from ..scales.continuous import X_AESTHETICS, Y_AESTHETICS
from .base import Position
from .dodge import collide_setup, pos_dodge

__all__ = [
    "PositionJitter",
    "PositionJitterdodge",
    "position_jitter",
    "position_jitterdodge",
]

#: Aesthetics ``position_jitterdodge`` can dodge by (one must be mapped).
DODGE_AES = ("fill", "color", "colour", "linetype", "shape", "size", "alpha")


def _num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _resolution(data: pd.DataFrame, axis: str, scales: dict[str, Any] | None) -> float:
    """ggplot2's ``resolution(x, zero = FALSE, discrete = TRUE)``: 1 on a discrete
    scale, else the smallest gap between distinct values (1 if fewer than two)."""
    scale = (scales or {}).get(axis)
    if getattr(scale, "is_discrete", False) or axis not in data.columns:
        return 1.0
    x = _num(data[axis])
    u = np.unique(x[np.isfinite(x)])
    if len(u) < 2:
        return 1.0
    d = np.diff(u)
    d = d[d > 1e-9 * max(1.0, float(np.abs(u).max()))]
    return float(d.min()) if len(d) else 1.0


def _auto_seed(data: pd.DataFrame) -> int:
    """Deterministic seed from the layer's x/y values (same data -> same noise)."""
    h = zlib.crc32(np.int64(len(data)).tobytes())
    for c in ("x", "y"):
        if c in data.columns:
            h = zlib.crc32(np.ascontiguousarray(_num(data[c])).tobytes(), h)
    return h


def jitter_offsets(n: int, width: float, height: float, seed: int) -> tuple[np.ndarray, ...]:
    """``(dx, dy)``: ``n`` uniform offsets in ``[-width, width]`` and
    ``[-height, height]`` (x drawn first, then y, from one generator)."""
    rng = np.random.default_rng(seed)
    dx = rng.uniform(-width, width, n) if width > 0 else np.zeros(n)
    dy = rng.uniform(-height, height, n) if height > 0 else np.zeros(n)
    return dx, dy


def shift_positions(data: pd.DataFrame, dx: Any = 0.0, dy: Any = 0.0) -> pd.DataFrame:
    """Add ``dx`` to every x-like column and ``dy`` to every y-like column."""
    data = data.copy()
    for cols, d in ((X_AESTHETICS, dx), (Y_AESTHETICS, dy)):
        if np.all(np.asarray(d) == 0):
            continue
        for c in cols:
            if c in data.columns:
                data[c] = _num(data[c]) + d
    return data


class PositionJitter(Position):
    """Jitter points to reduce overplotting (see the module docstring)."""

    name = "jitter"

    def __init__(
        self, width: float | None = None, height: float | None = None, seed: int | None = None
    ) -> None:
        super().__init__(width=width, height=height, seed=seed)

    def compute_layer(
        self, data: pd.DataFrame, params: dict[str, Any], scales: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        if len(data) == 0:
            return data
        width = params.get("width")
        height = params.get("height")
        if width is None:
            width = 0.4 * _resolution(data, "x", scales)
        if height is None:
            height = 0.4 * _resolution(data, "y", scales)
        seed = params.get("seed")
        seed = _auto_seed(data) if seed is None else seed
        dx, dy = jitter_offsets(len(data), float(width), float(height), seed)
        for d, col in ((dx, "x"), (dy, "y")):
            if col in data.columns:
                d[~np.isfinite(_num(data[col]))] = 0.0
        return shift_positions(data, dx, dy)

    def compute_panel(
        self, data: pd.DataFrame, params: dict[str, Any], scales: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        return self.compute_layer(data, params, scales)


class PositionJitterdodge(Position):
    """Dodge, then jitter within each dodged slot (``position_jitterdodge``)."""

    name = "jitterdodge"

    def __init__(
        self,
        jitter_width: float | None = None,
        jitter_height: float = 0.0,
        dodge_width: float = 0.75,
        reverse: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            jitter_width=jitter_width,
            jitter_height=jitter_height,
            dodge_width=dodge_width,
            reverse=reverse,
            seed=seed,
        )

    def setup_params(self, data: pd.DataFrame) -> dict[str, Any]:
        params = dict(self.params)
        if len(data) and not any(a in data.columns for a in DODGE_AES):
            raise ValueError(
                "position_jitterdodge() requires at least one aesthetic to dodge by "
                "(fill, colour, linetype, shape, size or alpha)"
            )
        ndodge = 1
        if len(data) and "group" in data.columns:
            keys = [data[c] for c in ("PANEL",) if c in data.columns] + [
                pd.Series(_num(data["x"]), index=data.index)
            ]
            ndodge = int(data["group"].groupby(keys, dropna=False).nunique().max())
        params["ndodge"] = ndodge
        return params

    def compute_layer(
        self, data: pd.DataFrame, params: dict[str, Any], scales: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        if len(data) == 0:
            return data
        width = params.get("jitter_width")
        if width is None:
            width = 0.4 * _resolution(data, "x", scales)
        params = {**params, "_jitter": float(width) / (params["ndodge"] + 2)}
        return super().compute_layer(data, params, scales)

    def compute_panel(
        self, data: pd.DataFrame, params: dict[str, Any], scales: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        index = data.index
        work, width = collide_setup(
            data.reset_index(drop=True), params["dodge_width"], params["reverse"]
        )
        key = _num(work["xmin"])
        pieces = [pos_dodge(p, width) for _, p in work.groupby(key, sort=False, dropna=False)]
        work = pd.concat(pieces).sort_index()
        seed = params.get("seed")
        seed = _auto_seed(data) if seed is None else seed
        dx, dy = jitter_offsets(len(work), params["_jitter"], float(params["jitter_height"]), seed)
        out = shift_positions(work, dx, dy)
        out.index = index
        return out


def position_jitter(
    width: float | None = None, height: float | None = None, seed: int | None = None
) -> PositionJitter:
    """Jitter points (ggplot2's ``position_jitter``).

    Parameters
    ----------
    width, height : float, optional
        Amount of jitter in each direction (the noise is in ``[-width, width]``),
        in transformed units. Default: 40% of the data's resolution (``0.4`` on a
        discrete axis).
    seed : int, optional
        Seed for :func:`numpy.random.default_rng`. ``None`` derives one from the
        data, so builds are reproducible either way.
    """
    return PositionJitter(width=width, height=height, seed=seed)


def position_jitterdodge(
    jitter_width: float | None = None,
    jitter_height: float = 0.0,
    dodge_width: float = 0.75,
    reverse: bool = False,
    seed: int | None = None,
) -> PositionJitterdodge:
    """Dodge by group, then jitter within each slot (ggplot2's
    ``position_jitterdodge``; ``jitter.width`` -> ``jitter_width`` etc.).

    Needs a mapped fill/colour/... to dodge by. The jitter width is
    ``jitter_width / (ndodge + 2)`` (default ``jitter_width`` = 40% of the x
    resolution), where ``ndodge`` is the largest number of groups at one x.
    """
    return PositionJitterdodge(
        jitter_width=jitter_width,
        jitter_height=jitter_height,
        dodge_width=dodge_width,
        reverse=reverse,
        seed=seed,
    )
