"""stat_density (for geom_density) and stat_ydensity (for geom_violin).

Both estimate densities with :func:`ggmanim.stats._kde.compute_density`, which
reproduces R's ``density()`` (Gaussian kernel by default, ``bw="nrd0"``,
``adjust``, ``kernel``, reflection at finite ``bounds``) and honour a ``weight``
aesthetic.

``stat_density`` -- per group, ``n = 512`` points over the x scale's range (or
the group's own range with ``trim=True``). Computed variables: ``density``,
``count`` (``density * n``), ``wdensity`` (``density * sum(weight)``),
``scaled``/``ndensity`` (``density / max``) and ``n``. ``geom_density`` maps
``y = after_stat("density")`` by default. ``orientation="y"`` estimates the
density of ``y`` (and maps ``x = after_stat("density")``).

``stat_ydensity`` -- per group (and per x level on a discrete x scale) the
density of ``y`` over the group's range (``trim=True``) or that range extended by
three bandwidths (``trim=False``); ``x`` stays the group's position. Adds
``violinwidth``, the half-width of the violin relative to its ``width``:
``scale="area"`` (every violin has the same area: ``density / max(density)``
over the panel), ``"count"`` (areas proportional to the number of observations)
or ``"width"`` (every violin as wide as ``width``: ``scaled``). ``quantiles``
(default ``(0.25, 0.5, 0.75)``) are inserted as extra rows with a ``quantile``
column (``NaN`` on the other rows), which ``geom_violin(draw_quantiles=...)``
draws.

Horizontal violins (``y`` discrete) are computed on swapped columns, see
:func:`ggmanim.stats.boxplot.resolve_flipped`.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from ..aes import after_stat
from ..registry import register
from ._kde import bandwidth, compute_density
from .base import Stat, scale_limits
from .boxplot import _finite_y, _is_discrete_scale, flip_names, regroup_by_x, resolve_flipped

__all__ = [
    "StatDensity",
    "StatYdensity",
    "density_mapping",
    "stat_density",
    "stat_ydensity",
]

_VIOLIN_SCALES = ("area", "count", "width")


def _weights(df: pd.DataFrame) -> np.ndarray | None:
    if "weight" not in df.columns:
        return None
    return pd.to_numeric(df["weight"], errors="coerce").to_numpy(dtype=float)


class StatDensity(Stat):
    """Kernel density estimate of ``x`` per group (see the module docstring)."""

    dropped_aes = ("weight",)

    def __init__(
        self,
        bw: Any = "nrd0",
        adjust: float = 1.0,
        kernel: str = "gaussian",
        n: int = 512,
        trim: bool = False,
        bounds: tuple[float, float] = (-np.inf, np.inf),
        orientation: str | None = None,
        na_rm: bool = False,
        **params,
    ):
        super().__init__(
            bw=bw,
            adjust=adjust,
            kernel=kernel,
            n=n,
            trim=trim,
            bounds=bounds,
            orientation=orientation,
            na_rm=na_rm,
            **params,
        )

    def compute_panel(self, df: pd.DataFrame, scales=None, **params) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        flipped = resolve_flipped(df, scales, params.get("orientation"), main_is_orthogonal=False)
        data = flip_names(df) if flipped else df.copy()
        if "x" not in data.columns:
            raise ValueError("stat_density requires an x aesthetic (or y with orientation='y')")
        data = data.drop(columns=["flipped_aes", "y"], errors="ignore")
        data = _finite_y(data.rename(columns={"x": "y"}), "stat_density", bool(params.get("na_rm")))
        data = data.rename(columns={"y": "x"})
        if data.empty:
            return data
        rng = scale_limits(scales, "y" if flipped else "x")
        if rng is None:
            x = pd.to_numeric(data["x"], errors="coerce").to_numpy(dtype=float)
            rng = (float(x.min()), float(x.max()))
        params = {**params, "flipped_aes": flipped, "range_": rng}
        out = super().compute_panel(data, scales, **params)
        return flip_names(out) if flipped else out

    def compute_group(  # type: ignore[override]
        self,
        df: pd.DataFrame,
        scales=None,
        bw: Any = "nrd0",
        adjust: float = 1.0,
        kernel: str = "gaussian",
        n: int = 512,
        trim: bool = False,
        bounds: tuple[float, float] = (-np.inf, np.inf),
        orientation: str | None = None,
        na_rm: bool = False,
        flipped_aes: bool = False,
        range_: tuple[float, float] | None = None,
    ) -> pd.DataFrame:
        x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
        if trim or range_ is None:
            lo, hi = float(x.min()), float(x.max())
        else:
            lo, hi = range_
        dens = compute_density(x, _weights(df), lo, hi, bw, adjust, kernel, int(n), bounds)
        if dens is None:
            return pd.DataFrame()
        out = pd.DataFrame(dens)
        out["flipped_aes"] = flipped_aes
        return out


class StatYdensity(Stat):
    """Density of ``y`` per group, for violins (see the module docstring)."""

    dropped_aes = ("weight",)

    def __init__(
        self,
        bw: Any = "nrd0",
        adjust: float = 1.0,
        kernel: str = "gaussian",
        trim: bool = True,
        scale: str = "area",
        drop: bool = True,
        bounds: tuple[float, float] = (-np.inf, np.inf),
        quantiles: Any = (0.25, 0.5, 0.75),
        orientation: str | None = None,
        na_rm: bool = False,
        **params,
    ):
        if scale not in _VIOLIN_SCALES:
            raise ValueError(f"scale must be one of {_VIOLIN_SCALES}, got {scale!r}")
        super().__init__(
            bw=bw,
            adjust=adjust,
            kernel=kernel,
            trim=trim,
            scale=scale,
            drop=drop,
            bounds=bounds,
            quantiles=quantiles,
            orientation=orientation,
            na_rm=na_rm,
            **params,
        )

    def compute_panel(self, df: pd.DataFrame, scales=None, **params) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        flipped = resolve_flipped(df, scales, params.get("orientation"))
        data = flip_names(df) if flipped else df.copy()
        if "y" not in data.columns:
            raise ValueError("stat_ydensity requires a y aesthetic (or x with orientation='y')")
        if "x" not in data.columns:
            data["x"] = 0.0
        data = _finite_y(data, "stat_ydensity", bool(params.get("na_rm")))
        if data.empty:
            return data
        if _is_discrete_scale(scales, "y" if flipped else "x"):
            data = regroup_by_x(data)
        data = data.drop(columns=["flipped_aes"], errors="ignore")
        scale = params.pop("scale", "area")
        if scale not in _VIOLIN_SCALES:
            raise ValueError(f"scale must be one of {_VIOLIN_SCALES}, got {scale!r}")
        out = super().compute_panel(data, scales, **params)
        if len(out) and "density" in out.columns:
            dens = pd.to_numeric(out["density"], errors="coerce").to_numpy(dtype=float)
            peak = np.nanmax(dens)
            if scale == "area":
                vw = dens / peak
            elif scale == "count":
                nn = pd.to_numeric(out["n"], errors="coerce").to_numpy(dtype=float)
                vw = dens / peak * nn / np.nanmax(nn)
            else:
                vw = pd.to_numeric(out["scaled"], errors="coerce").to_numpy(dtype=float)
            out["violinwidth"] = vw
        out["flipped_aes"] = flipped
        return flip_names(out) if flipped else out

    @classmethod
    def parameters(cls) -> set:
        return super().parameters() | {"scale"}

    def compute_group(  # type: ignore[override]
        self,
        df: pd.DataFrame,
        scales=None,
        bw: Any = "nrd0",
        adjust: float = 1.0,
        kernel: str = "gaussian",
        trim: bool = True,
        drop: bool = True,
        bounds: tuple[float, float] = (-np.inf, np.inf),
        quantiles: Any = (0.25, 0.5, 0.75),
        orientation: str | None = None,
        na_rm: bool = False,
    ) -> pd.DataFrame:
        y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
        x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
        if len(y) < 2:
            if drop:
                warnings.warn(
                    "stat_ydensity: groups with fewer than two data points have been dropped",
                    stacklevel=4,
                )
                return pd.DataFrame()
            return pd.DataFrame({"x": x, "n": len(y)})
        h = bandwidth(y, bw)
        mod = 0.0 if trim else 3.0
        lo, hi = float(y.min()) - mod * h, float(y.max()) + mod * h
        dens = compute_density(y, _weights(df), lo, hi, h, adjust, kernel, 512, bounds)
        if dens is None:
            return pd.DataFrame()
        out = pd.DataFrame(dens).rename(columns={"x": "y"})
        if len(np.unique(x)) > 1:
            out["x"] = (x.min() + x.max()) / 2
            out["width"] = float(np.ptp(x)) * 0.9
        else:
            out["x"] = x[0]
        if quantiles is not None and len(np.atleast_1d(quantiles)):
            probs = np.atleast_1d(np.asarray(quantiles, dtype=float))
            if ((probs < 0) | (probs > 1)).any():
                raise ValueError("quantiles must be between 0 and 1")
            qy = np.quantile(y, probs)
            grid = out["y"].to_numpy(dtype=float)
            q = pd.DataFrame({"y": qy, "quantile": probs})
            for col in out.columns:
                if col in q.columns:
                    continue
                v = out[col]
                if pd.api.types.is_numeric_dtype(v):
                    q[col] = np.interp(qy, grid, v.to_numpy(dtype=float))
                else:
                    q[col] = v.iloc[0]
            out = out[~np.isin(grid, qy)]
            out = pd.concat([out, q], ignore_index=True)
        return out


def _stat_ctor(stat: str, geom, mapping, data, position, inherit_aes, name, params):
    from ..geoms.count import stat_layer  # geoms import manim: keep stats free of it

    return stat_layer(
        stat,
        geom,
        mapping,
        data,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )


@register
def stat_density(
    mapping=None,
    data=None,
    *,
    geom="area",
    position="stack",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Kernel density estimate drawn with ``geom`` (default a stacked
    ``geom_area``, as in ggplot2); ``y = after_stat("density")`` unless mapped.
    Params: ``bw``, ``adjust``, ``kernel``, ``n``, ``trim``, ``bounds``. Returns a
    layer."""
    mapping = density_mapping(mapping, params)
    return _stat_ctor("density", geom, mapping, data, position, inherit_aes, name, params)


def density_mapping(mapping, params):
    """``mapping`` plus the density default: ``y = after_stat("density")`` (with
    ``orientation="y"``: ``x = after_stat("density")``) unless that aesthetic is
    mapped in ``mapping`` or set as a constant in ``params``."""
    from ..aes import Aes, standardise_aes_name

    if isinstance(mapping, pd.DataFrame):
        return mapping
    given = Aes(mapping) if mapping is not None else Aes()
    target = "x" if params.get("orientation") == "y" else "y"
    if target in given or target in {standardise_aes_name(k) for k in params}:
        return mapping
    return Aes({target: after_stat("density")}).merge(given)


@register
def stat_ydensity(
    mapping=None,
    data=None,
    *,
    geom="violin",
    position="dodge",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Violin densities of ``y`` per x drawn with ``geom`` (default
    ``geom_violin``); see :mod:`ggmanim.stats.density`. Returns a layer."""
    from ..geoms.boxplot import oriented

    return _stat_ctor(
        "ydensity", geom, mapping, data, oriented(position), inherit_aes, name, params
    )
