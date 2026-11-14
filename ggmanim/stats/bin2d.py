"""stat_bin_2d: counts of observations in rectangular 2-D bins (for geom_bin_2d).

Breaks per axis follow ggplot2 (>= 4.0, ``compute_bins``): ``breaks`` if given;
else multiples of ``binwidth`` aligned on ``boundary`` (default 0) or
``center``; else ``bins`` (default 30) bins, of width ``range / (bins - 1)``
(``range / bins`` when a range end falls on a bin edge), aligned on
``boundary`` (default 0). The range is the scale's (all layers). Bins are
right-closed (``(a, b]``, ``closed="left"`` for ``[a, b)``) and the outermost
edge is included. ``bins``, ``binwidth``, ``breaks``, ``boundary``, ``center``
and ``closed`` accept one value for both axes or an ``(x, y)`` pair.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..aes import after_stat
from ..registry import register
from .base import Stat, scale_limits
from .contour import _stat_layer

__all__ = ["StatBin2d", "bin2d_breaks", "stat_bin_2d", "stat_bin2d"]


def _breaks_width(lo: float, hi: float, width: float, boundary, center) -> np.ndarray:
    """ggplot2's ``bin_breaks_width``."""
    if not width > 0:
        raise ValueError(f"binwidth must be positive, got {width!r}")
    if boundary is None:
        boundary = width / 2 if center is None else center - width / 2
    origin = boundary + math.floor((lo - boundary) / width) * width
    max_x = hi + (1 - 1e-8) * width
    k = int(math.floor((max_x - origin) / width + 1e-10))
    out = origin + width * np.arange(k + 1)
    if len(out) == 1:
        out = np.array([out[0], out[0] + width])
    return out


def bin2d_breaks(
    lo: float,
    hi: float,
    breaks=None,
    binwidth: float | None = None,
    bins: int = 30,
    boundary: float | None = 0.0,
    center: float | None = None,
) -> np.ndarray:
    """Bin edges for the range ``[lo, hi]`` (ggplot2's ``compute_bins`` as used
    by ``stat_bin_2d``: ``boundary`` defaults to 0 unless ``center`` is given)."""
    if breaks is not None:
        return np.sort(np.asarray(breaks, dtype=float))
    if boundary is not None and center is not None:
        raise ValueError("Only one of boundary and center may be specified.")
    if binwidth is not None:
        return _breaks_width(lo, hi, float(binwidth), boundary, center)
    bins = int(bins)
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")
    if hi - lo == 0:
        width = 0.1
    elif bins == 1:
        width, boundary, center = hi - lo, lo, None
    else:
        width = (hi - lo) / (bins - 1)
        if center is None and boundary is None:
            boundary = lo - width / 2
        if boundary is not None and any(v % width == boundary % width for v in (lo, hi)):
            width = (hi - lo) / bins
    return _breaks_width(lo, hi, width, boundary, center)


def _fuzzy(breaks: np.ndarray, closed: str = "right") -> np.ndarray:
    """ggplot2's ``bins()`` fuzzy edges: 1e-8 of the median width outward at
    the closed end."""
    if len(breaks) < 2:
        return breaks
    d = 1e-8 * float(np.median(np.diff(breaks)))
    if closed == "right":
        return breaks + np.r_[-d, np.full(len(breaks) - 1, d)]
    return breaks + np.r_[np.full(len(breaks) - 1, -d), d]


def _per_axis(v: Any, axis: int, breaks: bool = False) -> Any:
    """``v`` for axis 0 (x) / 1 (y): a pair gives one per axis, a scalar both."""
    if v is None:
        return None
    if isinstance(v, (tuple, list)) and len(v) == 2:
        # a pair of break sequences, or a pair of scalars (not a 2-break vector)
        if all(np.ndim(e) == 1 for e in v) or all(np.ndim(e) == 0 and not breaks for e in v):
            return v[axis]
    if isinstance(v, dict):
        return v.get("xy"[axis])
    return v


def _bin_index(values: np.ndarray, breaks: np.ndarray, closed: str = "right") -> np.ndarray:
    """Bin of each value (R's ``cut(..., include.lowest=TRUE)`` on the fuzzy
    edges); -1 outside."""
    if closed not in ("right", "left"):
        raise ValueError(f"closed must be 'right' or 'left', got {closed!r}")
    f = _fuzzy(breaks, closed)
    nb = len(breaks) - 1
    with np.errstate(invalid="ignore"):
        if closed == "right":
            idx = np.searchsorted(f, values, side="left") - 1
            idx = np.where(values == f[0], 0, idx)
        else:
            idx = np.searchsorted(f, values, side="right") - 1
            idx = np.where(values == f[-1], nb - 1, idx)
    ok = np.isfinite(values) & (idx >= 0) & (idx < nb)
    return np.where(ok, idx, -1)


class StatBin2d(Stat):
    """Rectangular 2-D binning.

    Parameters: ``bins``, ``binwidth``, ``breaks``, ``boundary``, ``center``,
    ``closed`` (see module docstring) and ``drop`` (drop empty bins, default True).

    Output (one row per non-empty bin, or every bin with ``drop=False``):
    ``xmin``/``xmax``/``ymin``/``ymax``, centres ``x``/``y``, ``width``/``height``,
    ``count`` (or summed ``weight``), ``ncount`` (``count / max``), ``density``
    (``count / sum``) and ``ndensity`` (``density / max``). Map
    ``fill=after_stat("count")`` (the default) or another computed variable.
    """

    required_aes = ("x", "y")
    default_aes = {"fill": after_stat("count")}
    dropped_aes = ("weight",)

    def compute_group(  # type: ignore[override]
        self,
        df,
        scales=None,
        bins=30,
        binwidth=None,
        breaks=None,
        drop=True,
        boundary=None,
        center=None,
        closed="right",
    ) -> pd.DataFrame:
        x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
        w = (
            pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if "weight" in df.columns
            else np.ones(len(df))
        )
        edges = []
        for axis, (v, aes) in enumerate(((x, "x"), (y, "y"))):
            ctr = _per_axis(center, axis)
            bnd = _per_axis(boundary, axis)
            if bnd is None and ctr is None:
                bnd = 0.0  # stat_bin_2d aligns bins on 0 by default
            lim = scale_limits(scales, aes)
            fin = v[np.isfinite(v)]
            if lim is None:
                if not len(fin):
                    return pd.DataFrame()
                lim = (float(fin.min()), float(fin.max()))
            b = bin2d_breaks(
                lim[0],
                lim[1],
                _per_axis(breaks, axis, breaks=True),
                _per_axis(binwidth, axis),
                int(_per_axis(bins, axis) or 30),
                bnd,
                ctr,
            )
            edges.append(b)
        xb, yb = edges
        nxb, nyb = len(xb) - 1, len(yb) - 1
        if nxb < 1 or nyb < 1:
            return pd.DataFrame()
        xi = _bin_index(x, xb, _per_axis(closed, 0) or "right")
        yi = _bin_index(y, yb, _per_axis(closed, 1) or "right")
        ok = (xi >= 0) & (yi >= 0)
        counts = np.zeros((nyb, nxb))
        np.add.at(counts, (yi[ok], xi[ok]), w[ok])
        gj, gi = np.meshgrid(np.arange(nyb), np.arange(nxb), indexing="ij")
        jj, ii, c = gj.ravel(), gi.ravel(), counts.ravel()
        if drop:
            keep = c > 0
            jj, ii, c = jj[keep], ii[keep], c[keep]
        out = pd.DataFrame(
            {
                "xmin": xb[ii],
                "xmax": xb[ii + 1],
                "ymin": yb[jj],
                "ymax": yb[jj + 1],
            }
        )
        out["x"] = (out["xmin"] + out["xmax"]) / 2
        out["y"] = (out["ymin"] + out["ymax"]) / 2
        out["width"] = out["xmax"] - out["xmin"]
        out["height"] = out["ymax"] - out["ymin"]
        out["count"] = c
        total, top = c.sum(), c.max() if len(c) else 0.0
        out["ncount"] = c / top if top else np.nan
        out["density"] = c / total if total else np.nan
        dmax = out["density"].max() if len(out) else np.nan
        out["ndensity"] = out["density"] / dmax if dmax else np.nan
        return out


@register
def stat_bin_2d(
    mapping=None,
    data=None,
    *,
    geom="bin_2d",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Rectangular 2-D bin counts of ``aes(x, y)`` drawn as tiles."""
    return _stat_layer("bin_2d", geom, mapping, data, position, inherit_aes, name, params)


stat_bin2d = register(name="stat_bin2d")(stat_bin_2d)
