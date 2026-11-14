"""stat_summary / stat_summary_bin: summarise ``y`` at each ``x`` (or x bin).

Summaries run per group and per unique ``x`` (``stat_summary``) or per x bin
(``stat_summary_bin``), in transformed space like every stat (so on a log y
scale the mean is the mean of the logs, as in ggplot2).

The summary is chosen as in ggplot2:

* ``fun_data`` -- a function of the values returning ``y``/``ymin``/``ymax``
  (a dict, ``Series`` or one-row ``DataFrame``), or one of the names
  ``"mean_se"``, ``"mean_cl_normal"``, ``"mean_sdl"``, ``"median_hilow"``
  (:mod:`ggmanim.stats._summary_funs`);
* else ``fun`` / ``fun_min`` / ``fun_max`` -- functions (or numpy names such as
  ``"mean"``, ``"median"``, ``"min"``, ``"max"``) giving ``y``, ``ymin``, ``ymax``
  (any subset);
* else ``mean_se``.

``fun_args`` (a dict) is passed to every summary function. Columns that are
constant within an ``x`` bucket (colour, ``key`` ...) are carried over.

``stat_summary(orientation="y")`` swaps the roles of the axes: ``x`` is
summarised at each unique ``y`` and the output has ``x``/``xmin``/``xmax`` (plus
``flipped_aes = True``), which the interval geoms draw horizontally. With
``orientation=None`` (the default) the stat is flipped when the y scale is
discrete and the x scale is not (ggplot2's ``has_flipped_aes``), else not.

The ``stat_summary()`` / ``stat_summary_bin()`` constructors return a *layer*
(default geom ``pointrange``, looked up by name in the component registry), so
``ggplot(df, aes("g", "v")).stat_summary(fun_data="mean_cl_normal")`` works; a
geom constructor also accepts ``stat="summary"``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..registry import COMPONENTS, register
from ._summary_funs import SUMMARY_FUNS, mean_cl_normal, mean_sdl, mean_se, median_hilow
from .base import Stat, _carry_constant_columns, scale_limits

__all__ = [
    "StatSummary",
    "StatSummaryBin",
    "stat_summary",
    "stat_summary_bin",
    "summary_function",
    "mean_se",
    "mean_cl_normal",
    "mean_sdl",
    "median_hilow",
]
__toplevel__ = ["mean_se", "mean_cl_normal", "mean_sdl", "median_hilow"]

_NUMPY_FUNS: dict[str, Callable[..., Any]] = {
    "mean": np.mean,
    "median": np.median,
    "min": np.min,
    "max": np.max,
    "sum": np.sum,
    "sd": lambda x, **k: np.std(x, ddof=1, **k),
    "var": lambda x, **k: np.var(x, ddof=1, **k),
    "length": lambda x, **k: float(len(x)),
}


def _resolve(fun: Any, what: str) -> Callable[..., Any]:
    if callable(fun):
        return fun
    if isinstance(fun, str):
        table: Mapping[str, Callable[..., Any]] = (
            SUMMARY_FUNS if what == "fun_data" else _NUMPY_FUNS
        )
        if fun in table:
            return table[fun]
        raise ValueError(f"unknown {what} {fun!r}; available: {', '.join(sorted(table))}")
    raise TypeError(f"{what} must be a function or a name, got {fun!r}")


def _as_dict(res: Any) -> dict[str, float]:
    if isinstance(res, pd.DataFrame):
        if len(res) != 1:
            raise ValueError("fun_data must return a single row")
        res = res.iloc[0]
    if isinstance(res, pd.Series):
        res = res.to_dict()
    if not isinstance(res, Mapping):
        raise TypeError(f"fun_data must return a dict/Series/DataFrame, got {type(res).__name__}")
    return {str(k): float(v) for k, v in res.items()}


def summary_function(
    fun_data: Any = None,
    fun: Any = None,
    fun_max: Any = None,
    fun_min: Any = None,
    fun_args: Mapping[str, Any] | None = None,
) -> Callable[[np.ndarray], dict[str, float]]:
    """The summary used by :class:`StatSummary`: values -> ``{"y", "ymin", "ymax"}``
    (the keys the chosen functions produce); see the module docstring."""
    args = dict(fun_args or {})
    if fun_data is not None:
        fd = _resolve(fun_data, "fun_data")
        return lambda x: _as_dict(fd(x, **args))
    if fun is None and fun_min is None and fun_max is None:
        return lambda x: mean_se(x, **args)
    funs = {
        k: _resolve(f, k)
        for k, f in (("y", fun), ("ymin", fun_min), ("ymax", fun_max))
        if f is not None
    }

    def summarise(x: np.ndarray) -> dict[str, float]:
        if len(x) == 0:
            return {k: np.nan for k in funs}
        return {k: float(f(x, **args)) for k, f in funs.items()}

    return summarise


def _summarise_buckets(
    df: pd.DataFrame, keys: np.ndarray, summarise, extra: Callable[[Any], dict] | None = None
) -> pd.DataFrame:
    """One summary row per distinct ``keys`` value (sorted), with the bucket's
    constant columns carried over."""
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    rows = []
    for k in np.unique(keys[np.isfinite(keys)]):
        sel = keys == k
        vals = y[sel]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        res = pd.DataFrame([{"x": float(k), **(extra(k) if extra else {}), **summarise(vals)}])
        rows.append(_carry_constant_columns(df[sel], res, ("y", "ymin", "ymax")))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


_FLIP = {"x": "y", "y": "x", "ymin": "xmin", "ymax": "xmax"}


def _check_orientation(orientation: Any) -> None:
    if orientation not in (None, "x", "y"):
        raise ValueError(f"orientation must be 'x', 'y' or None, got {orientation!r}")


def _flipped(orientation: Any, scales) -> bool:
    """Whether to summarise ``x`` per ``y``: ``orientation == "y"``; with
    ``None``, when the y scale is discrete and the x scale is not."""
    _check_orientation(orientation)
    if orientation is not None:
        return orientation == "y"
    if not scales:
        return False
    sx, sy = scales.get("x"), scales.get("y")
    return bool(getattr(sy, "is_discrete", False)) and not getattr(sx, "is_discrete", False)


class StatSummary(Stat):
    """Summarise ``y`` per group at each unique ``x``. Output: ``x`` plus
    ``y``/``ymin``/``ymax`` (those the summary produces). With
    ``orientation="y"`` it summarises ``x`` per unique ``y`` instead (output
    ``y``, ``x``/``xmin``/``xmax`` and ``flipped_aes``); see the module
    docstring."""

    required_aes = ("x", "y")
    dropped_aes = ("y",)

    def __init__(
        self,
        fun_data=None,
        fun=None,
        fun_max=None,
        fun_min=None,
        fun_args=None,
        orientation=None,
        **params,
    ):
        _check_orientation(orientation)
        super().__init__(
            fun_data=fun_data,
            fun=fun,
            fun_max=fun_max,
            fun_min=fun_min,
            fun_args=fun_args,
            orientation=orientation,
            **params,
        )

    def compute_panel(self, df: pd.DataFrame, scales=None, **params) -> pd.DataFrame:
        params["orientation"] = "y" if _flipped(params.get("orientation"), scales) else "x"
        return super().compute_panel(df, scales, **params)

    def compute_group(  # type: ignore[override]
        self,
        df: pd.DataFrame,
        scales=None,
        fun_data=None,
        fun=None,
        fun_max=None,
        fun_min=None,
        fun_args=None,
        orientation=None,
    ) -> pd.DataFrame:
        summarise = summary_function(fun_data, fun, fun_max, fun_min, fun_args)
        flipped = orientation == "y"
        if flipped:
            df = df.rename(columns={"x": "y", "y": "x"})
        x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
        res = _summarise_buckets(df.drop(columns=["x"]), x, summarise)
        if flipped and len(res):
            res = res.rename(columns=_FLIP).assign(flipped_aes=True)
        return res


def bin_breaks(
    lo: float, hi: float, bins: int = 30, binwidth: float | None = None, breaks: Any = None
) -> np.ndarray:
    """Bin edges over ``[lo, hi]``: ``breaks`` as given, else ``binwidth``-wide
    bins starting at ``lo``, else ``bins`` equal bins."""
    if breaks is not None:
        return np.sort(np.asarray(breaks, dtype=float))
    if hi <= lo:
        hi = lo + 1.0
    if binwidth is not None:
        binwidth = float(binwidth)
        if binwidth <= 0:
            raise ValueError("binwidth must be positive")
        n = max(int(np.ceil((hi - lo) / binwidth - 1e-9)), 1)
        return lo + binwidth * np.arange(n + 1)
    bins = int(bins)
    if bins < 1:
        raise ValueError("bins must be >= 1")
    return np.linspace(lo, hi, bins + 1)


class StatSummaryBin(Stat):
    """Summarise ``y`` per group within x bins shared by the whole panel
    (``bins=30`` over the x scale's limits, or ``binwidth=``, or ``breaks=``;
    bins are left-closed, the last one also includes its right edge). Output:
    ``x`` (bin centre), ``width`` (bin width) and ``y``/``ymin``/``ymax``."""

    required_aes = ("x", "y")
    dropped_aes = ("y",)

    def __init__(
        self,
        fun_data=None,
        fun=None,
        fun_max=None,
        fun_min=None,
        fun_args=None,
        bins=30,
        binwidth=None,
        breaks=None,
        **params,
    ):
        super().__init__(
            fun_data=fun_data,
            fun=fun,
            fun_max=fun_max,
            fun_min=fun_min,
            fun_args=fun_args,
            bins=bins,
            binwidth=binwidth,
            breaks=breaks,
            **params,
        )

    def compute_panel(self, df: pd.DataFrame, scales=None, **params) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        lim = scale_limits(scales, "x")
        if lim is None:
            x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            lim = (float(x.min()), float(x.max())) if len(x) else (0.0, 1.0)
        params["breaks"] = bin_breaks(
            lim[0], lim[1], params.get("bins", 30), params.get("binwidth"), params.get("breaks")
        )
        return super().compute_panel(df, scales, **params)

    def compute_group(  # type: ignore[override]
        self,
        df: pd.DataFrame,
        scales=None,
        fun_data=None,
        fun=None,
        fun_max=None,
        fun_min=None,
        fun_args=None,
        bins=30,
        binwidth=None,
        breaks=None,
    ) -> pd.DataFrame:
        summarise = summary_function(fun_data, fun, fun_max, fun_min, fun_args)
        edges = np.asarray(
            breaks if breaks is not None else bin_breaks(0, 1, bins, binwidth), dtype=float
        )
        x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
        idx = np.searchsorted(edges, x, side="right") - 1
        idx = np.where(np.isclose(x, edges[-1]), len(edges) - 2, idx)
        ok = np.isfinite(x) & (idx >= 0) & (idx < len(edges) - 1)
        idx = np.where(ok, idx, -1)
        mids = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges)
        res = _summarise_buckets(
            df.drop(columns=["x"]),
            np.where(ok, idx, np.nan).astype(float),
            summarise,
            extra=lambda k: {"width": float(widths[int(k)])},
        )
        if len(res):
            res["x"] = mids[res["x"].to_numpy(dtype=int)]
        return res


def _summary_layer(
    stat: str,
    geom: Any,
    mapping,
    data,
    position,
    inherit_aes: bool,
    name: str | None,
    stat_params: dict[str, Any],
    params: dict[str, Any],
):
    if callable(geom):
        ctor = geom
    else:
        key = str(geom)
        key = key if key.startswith("geom_") else f"geom_{key}"
        if key not in COMPONENTS:
            importlib.import_module("ggmanim.geoms")  # registers the geoms (imports manim)
        ctor = COMPONENTS.get(key)
        if ctor is None:
            raise ValueError(f"unknown geom {geom!r} for {stat}")
    stat_params = {k: v for k, v in stat_params.items() if v is not None}
    return ctor(
        mapping,
        data,
        stat=stat,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        **stat_params,
        **params,
    )


@register
def stat_summary(
    mapping=None,
    data=None,
    *,
    geom: Any = "pointrange",
    position="identity",
    fun_data: Any = None,
    fun: Any = None,
    fun_max: Any = None,
    fun_min: Any = None,
    fun_args: Mapping[str, Any] | None = None,
    orientation: str | None = None,
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """A layer summarising ``y`` at each unique ``x`` (default ``mean_se``),
    drawn with ``geom`` (a name such as ``"pointrange"``, ``"errorbar"``,
    ``"crossbar"``, ``"point"``, ``"line"``, ``"col"``, or a ``geom_*``
    constructor). ``orientation="y"`` summarises ``x`` at each unique ``y``
    instead (horizontal intervals); ``None`` picks ``"y"`` when only the y axis
    is discrete. Other params (constant aesthetics, ``width=`` ...) go to the
    geom."""
    _check_orientation(orientation)
    sp = dict(
        fun_data=fun_data,
        fun=fun,
        fun_max=fun_max,
        fun_min=fun_min,
        fun_args=fun_args,
        orientation=orientation,
    )
    return _summary_layer("summary", geom, mapping, data, position, inherit_aes, name, sp, params)


@register
def stat_summary_bin(
    mapping=None,
    data=None,
    *,
    geom: Any = "pointrange",
    position="identity",
    fun_data: Any = None,
    fun: Any = None,
    fun_max: Any = None,
    fun_min: Any = None,
    fun_args: Mapping[str, Any] | None = None,
    bins: int = 30,
    binwidth: float | None = None,
    breaks: Any = None,
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Like :func:`stat_summary` but over x bins (``bins``, ``binwidth`` or
    ``breaks``, in transformed x units); ``x`` is the bin centre and ``width``
    the bin width (used by ``geom="crossbar"``/``"col"``/``"errorbar"``)."""
    sp = dict(
        fun_data=fun_data,
        fun=fun,
        fun_max=fun_max,
        fun_min=fun_min,
        fun_args=fun_args,
        bins=bins,
        binwidth=binwidth,
        breaks=breaks,
    )
    return _summary_layer(
        "summary_bin", geom, mapping, data, position, inherit_aes, name, sp, params
    )
