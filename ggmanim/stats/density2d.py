"""stat_density_2d / stat_density_2d_filled: 2-D kernel density + contours.

Per group, a bivariate Gaussian KDE is evaluated on an ``n x n`` grid spanning
the x and y scale limits (MASS's ``kde2d``, which ggplot2 uses):

* bandwidths ``h`` per axis default to :func:`bandwidth_nrd`
  (``4 * 1.06 * min(sd, IQR / 1.34) * n^(-1/5)``), times ``adjust``;
* like ``kde2d``, the kernel standard deviation is ``h / 4``.

With ``contour=True`` (the default) the grid is contoured (lines for
``stat_density_2d``, bands for ``stat_density_2d_filled``) on ``contour_var``
(``"density"``, ``"ndensity"`` or ``"count"``) with levels shared across the
groups of the layer (see :func:`~.contour.contour_breaks`). With
``contour=False`` the grid itself is returned (``x``, ``y``, ``density``,
``ndensity``, ``count``, ``n``), e.g. for ``geom_raster(aes(fill=after_stat("density")))``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..aes import after_stat
from ..registry import register
from .base import Stat, scale_limits
from .contour import _renumber, _stat_layer, _z_range, contour_bands, contour_lines

__all__ = [
    "StatDensity2d",
    "StatDensity2dFilled",
    "bandwidth_nrd",
    "kde2d",
    "stat_density_2d",
    "stat_density_2d_filled",
    "stat_density2d",
    "stat_density2d_filled",
]


def bandwidth_nrd(x) -> float:
    """MASS's ``bandwidth.nrd``: ``4 * 1.06 * min(sd, IQR / 1.34) * n^(-1/5)``
    (quartiles as R's default type 7)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    q1, q3 = np.quantile(x, [0.25, 0.75])
    return float(4 * 1.06 * min(np.std(x, ddof=1), (q3 - q1) / 1.34) * len(x) ** (-0.2))


def _pair(v, name: str) -> tuple[float, float]:
    arr = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    if len(arr) == 2:
        return float(arr[0]), float(arr[1])
    raise ValueError(f"{name} must be a number or a pair, got {v!r}")


def kde2d(
    x,
    y,
    h: float | Sequence[float] | None = None,
    n: int | Sequence[int] = 25,
    lims: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MASS's ``kde2d``: ``(gx, gy, z)`` with ``z[i, j]`` the density at
    ``(gx[i], gy[j])``.

    ``h`` (bandwidth per axis; default :func:`bandwidth_nrd` of each) is divided
    by 4 to give the Gaussian kernel's standard deviation. ``n`` grid points per
    axis (one number or a pair), spanning ``lims = (xlo, xhi, ylo, yhi)``
    (default: the data ranges)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx_, ny_ = (int(v) for v in _pair(n, "n"))
    if lims is None:
        lims = (x.min(), x.max(), y.min(), y.max())
    gx = np.linspace(lims[0], lims[1], nx_)
    gy = np.linspace(lims[2], lims[3], ny_)
    hx, hy = (bandwidth_nrd(x), bandwidth_nrd(y)) if h is None else _pair(h, "h")
    if not (hx > 0 and hy > 0):
        raise ValueError(f"bandwidths must be strictly positive, got h=({hx:g}, {hy:g})")
    hx, hy = hx / 4, hy / 4
    ax = (gx[:, None] - x[None, :]) / hx
    ay = (gy[:, None] - y[None, :]) / hy
    phi_x = np.exp(-0.5 * ax**2) / np.sqrt(2 * np.pi)
    phi_y = np.exp(-0.5 * ay**2) / np.sqrt(2 * np.pi)
    z = phi_x @ phi_y.T / (len(x) * hx * hy)
    return gx, gy, z


class StatDensity2d(Stat):
    """2-D kernel density estimate, contoured into lines (see module docstring).

    Parameters (``compute_group``): ``n`` (grid size, default 100), ``h``
    (bandwidth per axis), ``adjust`` (multiplier, number or pair), ``contour``,
    ``contour_var``, and the contour ``bins``/``binwidth``/``breaks``.
    """

    required_aes = ("x", "y")
    dropped_aes = ("weight",)
    contour_type = "lines"

    def compute_layer(self, df, scales=None, **params):
        params = self._filter_params(params)
        grid_params = {k: v for k, v in params.items() if k in ("n", "h", "adjust")}
        grid = super().compute_layer(df, scales, **grid_params)
        if not params.get("contour", True) or grid.empty:
            return grid
        var = params.get("contour_var", "density")
        if var not in ("density", "ndensity", "count"):
            raise ValueError(f"contour_var must be 'density', 'ndensity' or 'count', got {var!r}")
        z_range = _z_range(grid, var)
        fn = contour_bands if self.contour_type == "bands" else contour_lines
        keys = ["_li"] if self.contour_type == "bands" else ["_li", "piece"]
        drop = {"density", "ndensity", "count", "n"}
        pieces = []
        by = ["PANEL", "group"] if "PANEL" in grid.columns else ["group"]
        for _, gdf in grid.groupby(by, sort=True, observed=True):
            res = fn(
                gdf,
                var,
                z_range,
                params.get("bins"),
                params.get("binwidth"),
                params.get("breaks"),
            )
            if len(res) == 0:
                continue
            const = {
                c: gdf[c].iloc[0]
                for c in gdf.columns
                if c not in res.columns and c not in drop and gdf[c].nunique(dropna=False) == 1
            }
            pieces.append(res.assign(**const))
        if not pieces:
            return grid.iloc[0:0]
        out = pd.concat(pieces, ignore_index=True)
        if "PANEL" in out.columns:
            return pd.concat(
                [_renumber(p, keys) for _, p in out.groupby("PANEL", sort=True)],
                ignore_index=True,
            )
        return _renumber(out, keys)

    def compute_group(  # type: ignore[override]
        self,
        df,
        scales=None,
        n=100,
        h=None,
        adjust=1.0,
        contour=True,
        contour_var="density",
        bins=None,
        binwidth=None,
        breaks=None,
    ) -> pd.DataFrame:
        x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 2:
            return pd.DataFrame()
        hh = (bandwidth_nrd(x), bandwidth_nrd(y)) if h is None else _pair(h, "h")
        ax, ay = _pair(adjust, "adjust")
        hh = (hh[0] * ax, hh[1] * ay)
        xl = scale_limits(scales, "x") or (float(x.min()), float(x.max()))
        yl = scale_limits(scales, "y") or (float(y.min()), float(y.max()))
        gx, gy, z = kde2d(x, y, h=hh, n=n, lims=(*xl, *yl))
        density = z.ravel(order="F")  # x varies fastest (R's expand.grid)
        top = density.max()
        return pd.DataFrame(
            {
                "x": np.tile(gx, len(gy)),
                "y": np.repeat(gy, len(gx)),
                "density": density,
                "ndensity": density / top if top > 0 else np.nan,
                "count": len(x) * density,
                "n": len(x),
            }
        )


class StatDensity2dFilled(StatDensity2d):
    """Like :class:`StatDensity2d`, contoured into filled bands
    (``fill=after_stat("level")``)."""

    default_aes = {"fill": after_stat("level")}
    contour_type = "bands"


@register
def stat_density_2d(
    mapping=None,
    data=None,
    *,
    geom="density_2d",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Contours of a 2-D kernel density estimate of ``aes(x, y)``."""
    return _stat_layer("density_2d", geom, mapping, data, position, inherit_aes, name, params)


@register
def stat_density_2d_filled(
    mapping=None,
    data=None,
    *,
    geom="density_2d_filled",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Filled contour bands of a 2-D kernel density estimate of ``aes(x, y)``."""
    return _stat_layer(
        "density_2d_filled", geom, mapping, data, position, inherit_aes, name, params
    )


stat_density2d = register(name="stat_density2d")(stat_density_2d)
stat_density2d_filled = register(name="stat_density2d_filled")(stat_density_2d_filled)
