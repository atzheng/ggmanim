"""stat_summary_hex: a summary of ``z`` in hexagonal bins (for geom_hex).

The hexagons are those of :mod:`.hex` (``bins`` or ``binwidth``, one value or
an ``(x, y)`` pair; bin widths from the scale ranges, lattice origin from the
data range). In each non-empty hexagon ``fun`` (default ``"mean"``; a callable
or a numpy name) summarises the finite ``z`` values, called as
``fun(values, **fun_args)``. The output ``value`` is mapped to ``fill`` by
default and drawn with ``geom_hex``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..aes import after_stat
from ..registry import register
from .base import Stat, scale_limits
from .bin2d import _per_axis
from .contour import _stat_layer
from .hex import _bounds
from .summary2d import summarise_z

__all__ = ["StatSummaryHex", "hex_cells", "stat_summary_hex"]

_SQRT3 = math.sqrt(3.0)


def hex_cells(x, y, binwidth: tuple[float, float], xlim, ylim) -> tuple[np.ndarray, np.ndarray]:
    """Lattice coordinates ``(ci, cj)`` of the hexagon holding each point, on the
    lattice of :func:`~.hex.hexbin` (centre ``(xlo + ci * wx, ylo + cj * sqrt(3) * wy)``)."""
    wx, wy = (float(v) for v in binwidth)
    if not (wx > 0 and wy > 0):
        raise ValueError(f"binwidth must be positive, got {binwidth!r}")
    xlo = _bounds(float(xlim[0]), float(xlim[1]), wx)[0]
    ylo = _bounds(float(ylim[0]), float(ylim[1]), wy)[0]
    sx = (np.asarray(x, dtype=float) - xlo) / wx
    sy = (np.asarray(y, dtype=float) - ylo) / (_SQRT3 * wy)
    i1, j1 = np.floor(sx + 0.5), np.floor(sy + 0.5)
    i2, j2 = np.floor(sx) + 0.5, np.floor(sy) + 0.5
    first = (sx - i1) ** 2 + 3 * (sy - j1) ** 2 <= (sx - i2) ** 2 + 3 * (sy - j2) ** 2
    return np.where(first, i1, i2), np.where(first, j1, j2)


class StatSummaryHex(Stat):
    """Summary of ``z`` in hexagonal bins.

    Parameters: ``bins`` (30), ``binwidth``, ``fun`` (default ``"mean"``),
    ``fun_args`` (dict) and ``drop`` (default True: drop hexagons whose value is
    missing).

    Output (one row per non-empty hexagon, sorted by centre y then x): centres
    ``x``/``y``, ``width``/``height`` (the bin widths) and ``value``.
    """

    required_aes = ("x", "y", "z")
    default_aes = {"fill": after_stat("value")}
    dropped_aes = ("z",)

    def compute_group(  # type: ignore[override]
        self, df, scales=None, bins=30, binwidth=None, drop=True, fun="mean", fun_args=None
    ) -> pd.DataFrame:
        x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
        z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
        fin = np.isfinite(x) & np.isfinite(y)
        if not fin.any():
            return pd.DataFrame()
        x, y, z = x[fin], y[fin], z[fin]
        widths = []
        for axis, (v, aes) in enumerate(((x, "x"), (y, "y"))):
            bw = _per_axis(binwidth, axis)
            if bw is None:
                lim = scale_limits(scales, aes) or (float(v.min()), float(v.max()))
                span = lim[1] - lim[0]
                bw = span / int(_per_axis(bins, axis) or 30) if span > 0 else 1.0
            widths.append(float(bw))
        ci, cj = hex_cells(  # origin: data range, as stat_bin_hex
            x, y, (widths[0], widths[1]), (x.min(), x.max()), (y.min(), y.max())
        )
        ci_i = np.rint(ci * 2).astype(np.int64)  # half-integer lattice -> integers
        cj_i = np.rint(cj * 2).astype(np.int64)
        cmin_i, cmin_j = ci_i.min(), cj_i.min()
        span_i = int(ci_i.max() - cmin_i) + 1
        cells = (cj_i - cmin_j) * span_i + (ci_i - cmin_i)
        ids, vals = summarise_z(z, cells, fun, fun_args)
        if drop:
            keep = np.isfinite(vals)
            ids, vals = ids[keep], vals[keep]
        xlo = _bounds(float(x.min()), float(x.max()), widths[0])[0]
        ylo = _bounds(float(y.min()), float(y.max()), widths[1])[0]
        gi = (ids % span_i + cmin_i) / 2
        gj = (ids // span_i + cmin_j) / 2
        return pd.DataFrame(
            {
                "x": xlo + gi * widths[0],
                "y": ylo + gj * _SQRT3 * widths[1],
                "width": widths[0],
                "height": widths[1],
                "value": vals,
            }
        )


@register
def stat_summary_hex(
    mapping=None,
    data=None,
    *,
    geom="hex",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """``fun`` (default mean) of ``aes(z=)`` in hexagonal bins of ``aes(x, y)``,
    drawn as hexagons with ``fill=after_stat("value")``."""
    return _stat_layer("summary_hex", geom, mapping, data, position, inherit_aes, name, params)
