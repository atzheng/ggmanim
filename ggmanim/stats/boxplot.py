"""stat_boxplot: five-number summaries, outliers and notches (ggplot2 semantics).

Per box (one per group; on a *discrete* x scale one per group and x level, so
``aes("treatment", "value")`` gives one box per treatment as in ggplot2) the
output row has

* ``ymin``, ``lower``, ``middle``, ``upper``, ``ymax`` -- R's ``quantile(type = 7)``
  at 0, .25, .5, .75, 1 (``numpy.quantile``'s default); when some values lie more
  than ``coef * IQR`` (``coef = 1.5``) beyond the hinges, the whiskers
  ``ymin``/``ymax`` end at the most extreme value within that range instead;
* ``outliers`` -- a list column with the values beyond the whiskers' range;
* ``notchlower``/``notchupper`` -- ``middle -/+ 1.58 * IQR / sqrt(n)``;
* ``x``, ``width`` (the ``width`` param, default ``0.75 * resolution(x)``: ``0.75``
  on a discrete scale; ``0.9 * range(x)`` for a group spanning several x values),
  ``relvarwidth = sqrt(n)``, ``flipped_aes`` and ``n``.

A ``weight`` aesthetic gives weighted quantiles (ggplot2 fits ``quantreg::rq``;
here the equivalent lower weighted quantile is used: the smallest value whose
cumulative weight reaches the probability) and ``n`` = the summed weight.

Orientation: horizontal boxes (``y`` discrete, ``x`` continuous, or
``orientation="y"``) are computed on swapped columns and come back with
``xmin``/``xlower``/``xmiddle``/``xupper``/``xmax`` and ``flipped_aes = True``.
The orientation helpers here (:func:`flip_names`, :func:`resolve_flipped`) are
shared with :mod:`ggmanim.stats.density` and the distribution geoms.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..registry import register
from .base import Stat

__all__ = ["StatBoxplot", "stat_boxplot", "flip_names", "resolve_flipped"]

_PAIRS = (
    ("x", "y"),
    ("xmin", "ymin"),
    ("xmax", "ymax"),
    ("xend", "yend"),
    ("xlower", "lower"),
    ("xmiddle", "middle"),
    ("xupper", "upper"),
    ("xmin_final", "ymin_final"),
    ("xmax_final", "ymax_final"),
    ("xintercept", "yintercept"),
)
_SWAP = {**{a: b for a, b in _PAIRS}, **{b: a for a, b in _PAIRS}}


def flip_names(data: pd.DataFrame) -> pd.DataFrame:
    """``data`` with x-like and y-like columns swapped (ggplot2's ``flip_data``):
    ``x <-> y``, ``xmin <-> ymin`` ..., and the boxplot names
    ``lower <-> xlower``, ``middle <-> xmiddle``, ``upper <-> xupper``."""
    return data.rename(columns=_SWAP)


def _is_discrete_scale(scales: Mapping[str, Any] | None, axis: str) -> bool:
    return bool(getattr((scales or {}).get(axis), "is_discrete", False))


def resolve_flipped(
    data: pd.DataFrame,
    scales: Mapping[str, Any] | None = None,
    orientation: str | None = None,
    main_is_orthogonal: bool = True,
) -> bool:
    """Whether a layer is horizontal (ggplot2's ``has_flipped_aes``, simplified).

    ``orientation="x"``/``"y"`` decides. Else, for boxplots and violins
    (``main_is_orthogonal``): ``y`` only -> vertical, ``x`` only -> horizontal;
    both -> horizontal when the y scale is discrete and the x scale is not. For
    densities (``main_is_orthogonal=False``): ``y`` without ``x`` -> horizontal.
    A ``flipped_aes`` column (written by an earlier step) is used as is."""
    if orientation is not None:
        if orientation not in ("x", "y"):
            raise ValueError(f"orientation must be 'x' or 'y', got {orientation!r}")
        return orientation == "y"
    if "flipped_aes" in data.columns:
        return bool(data["flipped_aes"].iloc[0]) if len(data) else False
    has_x, has_y = "x" in data.columns, "y" in data.columns
    if not main_is_orthogonal:
        return has_y and not has_x
    if has_x and not has_y:
        return True
    if has_x and has_y:
        return _is_discrete_scale(scales, "y") and not _is_discrete_scale(scales, "x")
    return False


def resolution(x: Any, zero: bool = True) -> float:
    """ggplot2's ``resolution``: the smallest gap between distinct values of
    ``x`` (and 0 when ``zero``); 1 when ``x`` has a single distinct value."""
    v = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0 or np.ptp(v) <= 1e-9 * max(1.0, float(np.abs(v).max())):
        return 1.0
    if zero:
        v = np.append(v, 0.0)
    d = np.diff(np.unique(v))
    d = d[d > np.sqrt(np.finfo(float).eps)]
    return float(d.min()) if len(d) else 1.0


def regroup_by_x(df: pd.DataFrame) -> pd.DataFrame:
    """Split groups by ``x`` (for a discrete x, as ggplot2 groups by every
    discrete aesthetic): ``group`` renumbered ``1..k`` in order of ``(x, group)``."""
    if df.empty:
        return df
    g = pd.to_numeric(df["group"], errors="coerce").to_numpy(dtype=float)
    x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
    keys = pd.DataFrame({"x": x, "g": g})
    ids = keys.groupby(["x", "g"], sort=True, dropna=False).ngroup().to_numpy(dtype="int64") + 1
    return df.assign(group=ids)


def _finite_y(df: pd.DataFrame, stat: str, na_rm: bool) -> pd.DataFrame:
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y)
    if "weight" in df.columns:
        ok &= np.isfinite(pd.to_numeric(df["weight"], errors="coerce").to_numpy(dtype=float))
    n = int((~ok).sum())
    if n and not na_rm:
        warnings.warn(
            f"{stat}: removed {n} row{'s' if n != 1 else ''} containing non-finite values",
            stacklevel=4,
        )
    return df[ok] if n else df


def weighted_quantile(y: np.ndarray, w: np.ndarray, probs: Any) -> np.ndarray:
    """The smallest ``y`` whose cumulative (normalised) weight reaches each of
    ``probs`` (0 -> min, 1 -> max): the solution of the weighted check-loss
    minimisation ``quantreg::rq(y ~ 1, weights = w, tau = probs)`` uses."""
    order = np.argsort(y, kind="stable")
    ys, ws = y[order], np.clip(w[order], 0.0, None)
    cw = np.cumsum(ws)
    total = cw[-1]
    out = []
    for p in np.atleast_1d(np.asarray(probs, dtype=float)):
        if p <= 0:
            out.append(ys[ws > 0][0] if (ws > 0).any() else ys[0])
            continue
        i = int(np.searchsorted(cw, p * total - 1e-12 * total, side="left"))
        out.append(ys[min(i, len(ys) - 1)])
    return np.asarray(out, dtype=float)


def boxplot_stats(
    y: Any, weight: Any = None, coef: float = 1.5
) -> tuple[dict[str, float], np.ndarray, float]:
    """``({ymin, lower, middle, upper, ymax}, outliers, n)`` for the values ``y``
    (see the module docstring)."""
    y = np.asarray(y, dtype=float)
    qs = [0.0, 0.25, 0.5, 0.75, 1.0]
    if weight is None:
        stats = np.quantile(y, qs)  # R's quantile type 7
        n = float(len(y))
    else:
        w = np.asarray(weight, dtype=float)
        stats = weighted_quantile(y, w, qs)
        n = float(w.sum())
    iqr = stats[3] - stats[1]
    out = (y < stats[1] - coef * iqr) | (y > stats[3] + coef * iqr)
    if out.any():
        inner = np.concatenate([stats[1:4], y[~out]])
        stats[0], stats[4] = inner.min(), inner.max()
    names = ("ymin", "lower", "middle", "upper", "ymax")
    return dict(zip(names, (float(s) for s in stats))), y[out], n


class StatBoxplot(Stat):
    """Boxplot statistics per group (see the module docstring)."""

    dropped_aes = ("x", "y", "weight")

    def __init__(
        self,
        coef: float = 1.5,
        width: float | None = None,
        orientation: str | None = None,
        na_rm: bool = False,
        **params,
    ):
        super().__init__(coef=coef, width=width, orientation=orientation, na_rm=na_rm, **params)

    def compute_panel(self, df: pd.DataFrame, scales=None, **params) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        flipped = resolve_flipped(df, scales, params.get("orientation"))
        data = flip_names(df) if flipped else df.copy()
        if "y" not in data.columns:
            raise ValueError("stat_boxplot requires a y aesthetic (or x with orientation='y')")
        if "x" not in data.columns:
            data["x"] = 0.0
        data = _finite_y(data, "stat_boxplot", bool(params.get("na_rm")))
        if data.empty:
            return data
        x_axis = "y" if flipped else "x"
        discrete = _is_discrete_scale(scales, x_axis)
        if discrete:
            data = regroup_by_x(data)
        if params.get("width") is None:
            res = 1.0 if discrete else resolution(data["x"])
            params = {**params, "width": 0.75 * res}
        params["flipped_aes"] = flipped
        data = data.drop(columns=["flipped_aes"], errors="ignore")
        out = super().compute_panel(data, scales, **params)
        return flip_names(out) if flipped else out

    def compute_group(  # type: ignore[override]
        self,
        df: pd.DataFrame,
        scales=None,
        coef: float = 1.5,
        width: float | None = None,
        orientation: str | None = None,
        na_rm: bool = False,
        flipped_aes: bool = False,
    ) -> pd.DataFrame:
        y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
        w = (
            pd.to_numeric(df["weight"], errors="coerce").to_numpy(dtype=float)
            if "weight" in df.columns
            else None
        )
        stats, outliers, n = boxplot_stats(y, w, float(coef))
        x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
        if "width" in df.columns and len(df):
            width = float(pd.to_numeric(df["width"], errors="coerce").iloc[0])
        elif len(np.unique(x)) > 1:
            width = float(np.ptp(x)) * 0.9
        iqr = stats["upper"] - stats["lower"]
        row: dict[str, Any] = dict(stats)
        row["outliers"] = [outliers]
        row["notchupper"] = stats["middle"] + 1.58 * iqr / np.sqrt(n)
        row["notchlower"] = stats["middle"] - 1.58 * iqr / np.sqrt(n)
        row["x"] = (float(np.min(x)) + float(np.max(x))) / 2
        row["width"] = width
        row["relvarwidth"] = np.sqrt(n)
        row["flipped_aes"] = flipped_aes
        row["n"] = n
        return pd.DataFrame({k: (v if isinstance(v, list) else [v]) for k, v in row.items()})


@register
def stat_boxplot(
    mapping=None,
    data=None,
    *,
    geom="boxplot",
    position="dodge2",
    coef: float = 1.5,
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Boxplot statistics drawn with ``geom`` (default ``geom_boxplot``); see
    :mod:`ggmanim.stats.boxplot`. Returns a layer."""
    from ..geoms.boxplot import oriented  # geoms import manim: keep stats free of it
    from ..geoms.count import stat_layer

    return stat_layer(
        "boxplot",
        geom,
        mapping,
        data,
        position=oriented(position),
        inherit_aes=inherit_aes,
        name=name,
        params={"coef": coef, **params},
    )
