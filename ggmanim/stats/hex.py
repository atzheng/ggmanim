"""stat_bin_hex: counts of observations in hexagonal bins (for geom_hex).

Geometry follows ggplot2 / the ``hexbin`` package. With bin widths ``(wx, wy)``
(default: scale range / ``bins`` per axis, ``bins=30``) and the data bounds
rounded out to multiples of the widths (``xlo``, ``ylo``), hexagon centres lie on two staggered lattices:

* ``(xlo + i * wx, ylo + j * sqrt(3) * wy)`` and
* ``(xlo + (i + 1/2) * wx, ylo + (j + 1/2) * sqrt(3) * wy)``.

In units of ``(wx, wy)`` this is a regular hexagonal lattice of pointy-topped
hexagons that are ``1`` wide (flat side to flat side), so the hexagons drawn by
:func:`hexagon_vertices` tessellate the plane. Each observation goes to the
nearest centre in those units (vectorised: the nearer of the two lattice
candidates).
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

__all__ = ["StatBinHex", "hexbin", "hexagon_vertices", "stat_bin_hex", "stat_binhex"]

_SQRT3 = math.sqrt(3.0)


def _bounds(lo: float, hi: float, w: float) -> tuple[float, float]:
    """ggplot2's ``hex_bounds``."""
    return math.floor(lo / w) * w - 1e-6, math.ceil(hi / w) * w + 1e-6


def hexbin(x, y, binwidth: tuple[float, float], weight=None, xlim=None, ylim=None) -> pd.DataFrame:
    """Hexagonal bin counts: ``x``, ``y`` (centres) and ``count`` (summed
    ``weight``) of every non-empty hexagon, sorted by centre (y, then x).

    ``xlim``/``ylim`` (default: the data ranges) set the lattice origin."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones(len(x)) if weight is None else np.asarray(weight, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, w = x[ok], y[ok], w[ok]
    if not len(x):
        return pd.DataFrame({"x": [], "y": [], "count": []})
    wx, wy = (float(v) for v in binwidth)
    if not (wx > 0 and wy > 0):
        raise ValueError(f"binwidth must be positive, got {binwidth!r}")
    xl = xlim if xlim is not None else (x.min(), x.max())
    yl = ylim if ylim is not None else (y.min(), y.max())
    xlo = _bounds(float(xl[0]), float(xl[1]), wx)[0]
    ylo = _bounds(float(yl[0]), float(yl[1]), wy)[0]
    sx = (x - xlo) / wx  # lattice-1 centres at integers
    sy = (y - ylo) / (_SQRT3 * wy)
    i1, j1 = np.floor(sx + 0.5), np.floor(sy + 0.5)
    i2, j2 = np.floor(sx) + 0.5, np.floor(sy) + 0.5
    d1 = (sx - i1) ** 2 + 3 * (sy - j1) ** 2
    d2 = (sx - i2) ** 2 + 3 * (sy - j2) ** 2
    first = d1 <= d2
    ci = np.where(first, i1, i2)
    cj = np.where(first, j1, j2)
    cells = pd.DataFrame({"cj": cj, "ci": ci, "count": w})
    out = cells.groupby(["cj", "ci"], sort=True)["count"].sum().reset_index()
    return pd.DataFrame(
        {
            "x": xlo + out["ci"].to_numpy() * wx,
            "y": ylo + out["cj"].to_numpy() * _SQRT3 * wy,
            "count": out["count"].to_numpy(dtype=float),
        }
    )


def hexagon_vertices(x, y, width, height) -> tuple[np.ndarray, np.ndarray]:
    """``(n, 6)`` vertex coordinates of the pointy-topped hexagons centred on
    ``(x, y)`` with bin widths ``width``/``height`` (counter-clockwise from the
    upper right). Neighbouring hexagons of one lattice share their edges."""
    x, y, width, height = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (x, y, width, height))
    )
    ang = np.deg2rad(30.0 + 60.0 * np.arange(6))
    r = 1.0 / _SQRT3  # circumradius of a hexagon 1 wide
    dx = r * np.cos(ang)  # +-1/2 or 0
    dy = r * np.sin(ang)
    return (
        x.ravel()[:, None] + width.ravel()[:, None] * dx[None, :],
        y.ravel()[:, None] + height.ravel()[:, None] * dy[None, :],
    )


class StatBinHex(Stat):
    """Hexagonal 2-D binning (see module docstring).

    Output (one row per non-empty hexagon): centres ``x``/``y``, ``width``/
    ``height`` (the bin widths, transformed units), ``count`` (or summed
    ``weight``), ``ncount``, ``density`` (``count / sum``) and ``ndensity``.
    """

    required_aes = ("x", "y")
    default_aes = {"fill": after_stat("count")}
    dropped_aes = ("weight",)

    def compute_group(  # type: ignore[override]
        self, df, scales=None, bins=30, binwidth=None
    ) -> pd.DataFrame:
        x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
        fin = np.isfinite(x) & np.isfinite(y)
        if not fin.any():
            return pd.DataFrame()
        lims = []
        for v, aes in ((x[fin], "x"), (y[fin], "y")):
            lim = scale_limits(scales, aes) or (float(v.min()), float(v.max()))
            lims.append(lim)
        widths = []
        for axis in (0, 1):
            bw = _per_axis(binwidth, axis)
            if bw is None:
                nb = int(_per_axis(bins, axis) or 30)
                span = lims[axis][1] - lims[axis][0]
                bw = span / nb if span > 0 else 1.0
            widths.append(float(bw))
        weight = (
            pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if "weight" in df.columns
            else None
        )
        out = hexbin(x, y, (widths[0], widths[1]), weight)  # origin: data range (ggplot2)
        c = out["count"].to_numpy(dtype=float)
        out["width"] = widths[0]
        out["height"] = widths[1]
        top, total = (c.max(), c.sum()) if len(c) else (0.0, 0.0)
        out["ncount"] = c / top if top else np.nan
        out["density"] = c / total if total else np.nan
        dmax = out["density"].max() if len(out) else np.nan
        out["ndensity"] = out["density"] / dmax if dmax else np.nan
        return out


@register
def stat_bin_hex(
    mapping=None,
    data=None,
    *,
    geom="hex",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Hexagonal 2-D bin counts of ``aes(x, y)`` drawn as hexagons."""
    return _stat_layer("bin_hex", geom, mapping, data, position, inherit_aes, name, params)


stat_binhex = register(name="stat_binhex")(stat_bin_hex)
