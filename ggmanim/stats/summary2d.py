"""stat_summary_2d: a summary of ``z`` in rectangular 2-D bins (for geom_tile).

The bins are those of :mod:`.bin2d` (``bins``, ``binwidth``, ``breaks``,
``boundary``, ``center``, ``closed``; one value or an ``(x, y)`` pair, the
range the scale's). In each bin ``fun`` (default ``"mean"``; a callable or a
numpy name such as ``"median"``, ``"sd"``, ``"length"``) summarises the finite
``z`` values, called as ``fun(values, **fun_args)``. Like ggplot2, the output
``value`` is mapped to ``fill`` by default and drawn with ``geom_tile``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..aes import after_stat
from ..registry import register
from .base import Stat, scale_limits
from .bin2d import _bin_index, _per_axis, bin2d_breaks
from .contour import _stat_layer
from .summary import _resolve

__all__ = ["StatSummary2d", "stat_summary_2d", "stat_summary2d"]


def summarise_z(
    z: np.ndarray, cells: np.ndarray, fun: Any = "mean", fun_args: Mapping[str, Any] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``(cell ids, values)``: ``fun`` of the finite ``z`` of each distinct
    non-negative ``cells`` id (sorted). Shared by :mod:`.summary_hex`."""
    f: Callable[..., Any] = _resolve(fun, "fun")
    args = dict(fun_args or {})
    ok = (cells >= 0) & np.isfinite(z)
    ids = np.unique(cells[ok])
    vals = np.array([float(f(z[ok][cells[ok] == i], **args)) for i in ids], dtype=float)
    return ids, vals


class StatSummary2d(Stat):
    """Summary of ``z`` in rectangular 2-D bins.

    Parameters: the binning of :class:`~.bin2d.StatBin2d`, ``fun`` (default
    ``"mean"``), ``fun_args`` (dict) and ``drop`` (default True: drop bins whose
    value is missing; ``False`` keeps every bin, empty ones with ``value`` NaN).

    Output: ``xmin``/``xmax``/``ymin``/``ymax``, centres ``x``/``y``,
    ``width``/``height`` and ``value``.
    """

    required_aes = ("x", "y", "z")
    default_aes = {"fill": after_stat("value")}
    dropped_aes = ("z",)

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
        fun="mean",
        fun_args=None,
    ) -> pd.DataFrame:
        x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
        z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
        edges = []
        for axis, (v, aes) in enumerate(((x, "x"), (y, "y"))):
            ctr = _per_axis(center, axis)
            bnd = _per_axis(boundary, axis)
            if bnd is None and ctr is None:
                bnd = 0.0
            lim = scale_limits(scales, aes)
            fin = v[np.isfinite(v)]
            if lim is None:
                if not len(fin):
                    return pd.DataFrame()
                lim = (float(fin.min()), float(fin.max()))
            edges.append(
                bin2d_breaks(
                    lim[0],
                    lim[1],
                    _per_axis(breaks, axis, breaks=True),
                    _per_axis(binwidth, axis),
                    int(_per_axis(bins, axis) or 30),
                    bnd,
                    ctr,
                )
            )
        xb, yb = edges
        nxb, nyb = len(xb) - 1, len(yb) - 1
        if nxb < 1 or nyb < 1:
            return pd.DataFrame()
        xi = _bin_index(x, xb, _per_axis(closed, 0) or "right")
        yi = _bin_index(y, yb, _per_axis(closed, 1) or "right")
        cells = np.where((xi >= 0) & (yi >= 0), yi * nxb + xi, -1)
        ids, vals = summarise_z(z, cells, fun, fun_args)
        if drop:
            keep = np.isfinite(vals)
            ids, vals = ids[keep], vals[keep]
        else:
            full = np.full(nxb * nyb, np.nan)
            full[ids] = vals
            ids, vals = np.arange(nxb * nyb), full
        jj, ii = ids // nxb, ids % nxb
        out = pd.DataFrame({"xmin": xb[ii], "xmax": xb[ii + 1], "ymin": yb[jj], "ymax": yb[jj + 1]})
        out["x"] = (out["xmin"] + out["xmax"]) / 2
        out["y"] = (out["ymin"] + out["ymax"]) / 2
        out["width"] = out["xmax"] - out["xmin"]
        out["height"] = out["ymax"] - out["ymin"]
        out["value"] = vals
        return out


@register
def stat_summary_2d(
    mapping=None,
    data=None,
    *,
    geom="tile",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """``fun`` (default mean) of ``aes(z=)`` in rectangular bins of ``aes(x, y)``,
    drawn as tiles with ``fill=after_stat("value")``."""
    return _stat_layer("summary_2d", geom, mapping, data, position, inherit_aes, name, params)


stat_summary2d = register(name="stat_summary2d")(stat_summary_2d)
