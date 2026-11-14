"""stat_contour / stat_contour_filled: contour lines and bands of a gridded ``z``.

The layer data is a (possibly incomplete) rectilinear grid ``aes(x, y, z)``;
contours are traced with the numpy marching squares of
:mod:`._marching_squares`.

Levels (:func:`contour_breaks`, ggplot2's rules): ``breaks`` (a sequence, or a
function of the z range) wins; else ``bins`` (number of bins) or ``binwidth``;
else ``pretty(z_range, 10)``. The z range is taken over the whole layer (per
panel), so every group shares the same levels.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..aes import after_stat
from ..registry import register
from ._marching_squares import grid_from_xyz, isoband, isolines
from .base import Stat

__all__ = [
    "StatContour",
    "StatContourFilled",
    "contour_breaks",
    "isoband_labels",
    "stat_contour",
    "stat_contour_filled",
]


def _fullseq(lo: float, hi: float, size: float) -> np.ndarray:
    """scales::fullseq: multiples of ``size`` covering ``[lo, hi]``."""
    if hi - lo == 0:
        return np.array([lo - size / 2, hi + size / 2])
    start = math.floor(lo / size + 1e-10) * size
    stop = math.ceil(hi / size - 1e-10) * size
    k = int(round((stop - start) / size))
    return start + size * np.arange(k + 1)


def contour_breaks(
    z_range: Sequence[float],
    bins: int | None = None,
    binwidth: float | None = None,
    breaks: Sequence[float] | Callable | None = None,
) -> np.ndarray:
    """Contour levels for ``z_range`` (ggplot2's ``contour_breaks``).

    ``breaks`` (numbers) is returned as is; a callable ``breaks(z_range, binwidth)``
    replaces the full-sequence rule below. Without ``bins``/``binwidth`` the
    levels are ``pretty(z_range, 10)``. ``bins`` rounds the range out to a tenth
    of its leading digit and spaces ``bins + 1`` levels across it; ``binwidth``
    takes multiples of the width covering the range."""
    lo, hi = (float(v) for v in z_range)
    if breaks is not None and not callable(breaks):
        return np.asarray(breaks, dtype=float)
    seq = breaks if callable(breaks) else (lambda r, w: _fullseq(r[0], r[1], w))
    if bins is None and binwidth is None:
        # late import: the scales package pulls in manim (palettes); stats must not
        from ..scales.breaks import pretty

        return pretty(lo, hi, 10)
    if bins is not None:
        bins = int(bins)
        span = hi - lo
        if span > 0:
            accuracy = float(f"{span:.0e}") / 10  # signif(diff, 1) / 10
            lo = math.floor(lo / accuracy) * accuracy
            hi = math.ceil(hi / accuracy) * accuracy
        if bins == 1:
            return np.array([lo, hi])
        binwidth = (hi - lo) / (bins - 1)
        out = np.asarray(seq((lo, hi), binwidth), dtype=float)
        if len(out) < bins + 1:
            binwidth = (hi - lo) / bins
            out = np.asarray(seq((lo, hi), binwidth), dtype=float)
        return out
    return np.asarray(seq((lo, hi), float(binwidth)), dtype=float)  # type: ignore[arg-type]


def _format_sig(values: np.ndarray, digits: int = 3) -> list[str]:
    """R's ``format(x, digits=, trim=TRUE)`` for a vector (fixed notation): the
    fewest decimals that show every value to ``digits`` significant digits."""
    dec = 0
    for v in values:
        if not np.isfinite(v) or v == 0:
            continue
        mag = math.floor(math.log10(abs(v)))
        d = max(0, digits - 1 - mag)
        s = f"{round(v, d):.{d}f}".rstrip("0") if d else ""
        dec = max(dec, len(s.split(".")[1]) if "." in s else 0)
    return [f"{v:.{dec}f}" for v in values]


def isoband_labels(breaks: Sequence[float], digits: int = 3) -> list[str]:
    """Band labels ``"(lo, hi]"`` for consecutive ``breaks`` (ggplot2's
    ``pretty_isoband_levels``)."""
    b = np.asarray(breaks, dtype=float)
    lows, highs = _format_sig(b[:-1], digits), _format_sig(b[1:], digits)
    return [f"({a}, {c}]" for a, c in zip(lows, highs)]


def _z_range(df: pd.DataFrame, var: str) -> tuple[float, float] | None:
    if var not in df.columns:
        return None
    z = pd.to_numeric(df[var], errors="coerce").to_numpy(dtype=float)
    z = z[np.isfinite(z)]
    return (float(z.min()), float(z.max())) if len(z) else None


def _renumber(res: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Integer groups ``1..k`` from ``group`` x ``keys`` (sorted), keys dropped."""
    if res.empty:
        return res
    res = res.copy()
    res["group"] = res.groupby(["group", *keys], sort=True).ngroup().to_numpy("int64") + 1
    return res.drop(columns=[k for k in keys if k.startswith("_")])


class StatContour(Stat):
    """Contour lines of ``z`` over the ``(x, y)`` grid.

    Output: one row per vertex with ``x``, ``y``, ``level`` (the contour value),
    ``nlevel`` (``level / max(level)``), ``piece`` (index of the line within its
    level) and an integer ``group`` per line (original group x level x piece).
    Closed contours repeat their first vertex.
    """

    required_aes = ("x", "y", "z")
    dropped_aes = ("z", "weight")

    def compute_layer(self, df, scales=None, **params):
        params = dict(params)
        if params.get("z_range") is None:
            params["z_range"] = _z_range(df, "z")
        return super().compute_layer(df, scales, **params)

    def compute_panel(self, df, scales=None, **params):
        return _renumber(super().compute_panel(df, scales, **params), ["_li", "piece"])

    def compute_group(  # type: ignore[override]
        self, df, scales=None, bins=None, binwidth=None, breaks=None, z_range=None
    ) -> pd.DataFrame:
        return contour_lines(df, "z", z_range, bins, binwidth, breaks)


class StatContourFilled(Stat):
    """Filled contour bands of ``z`` over the ``(x, y)`` grid.

    Output: one row per ring vertex with ``x``, ``y``, ``level`` (an ordered
    categorical of band labels ``"(lo, hi]"``; every band is a category),
    ``level_low``, ``level_high``, ``level_mid``, ``nlevel``
    (``level_high / max``), ``subgroup`` (ring index within the band), ``piece``
    and an integer ``group`` per band. Rings are oriented (outer boundaries
    counter-clockwise, holes clockwise) so they fill with the nonzero rule; the
    ring of a region not bounded by a higher band carries no hole.
    """

    required_aes = ("x", "y", "z")
    default_aes = {"fill": after_stat("level")}
    dropped_aes = ("z", "weight")

    def compute_layer(self, df, scales=None, **params):
        params = dict(params)
        if params.get("z_range") is None:
            params["z_range"] = _z_range(df, "z")
        return super().compute_layer(df, scales, **params)

    def compute_panel(self, df, scales=None, **params):
        return _renumber(super().compute_panel(df, scales, **params), ["_li"])

    def compute_group(  # type: ignore[override]
        self, df, scales=None, bins=None, binwidth=None, breaks=None, z_range=None
    ) -> pd.DataFrame:
        return contour_bands(df, "z", z_range, bins, binwidth, breaks)


_LINE_COLS = ["x", "y", "level", "nlevel", "piece", "_li"]
_BAND_COLS = [
    "x",
    "y",
    "level",
    "level_low",
    "level_high",
    "level_mid",
    "nlevel",
    "subgroup",
    "piece",
    "_li",
]


def contour_lines(df, var, z_range, bins=None, binwidth=None, breaks=None) -> pd.DataFrame:
    """Contour-line rows (see :class:`StatContour`) of column ``var`` of ``df``."""
    z_range = z_range if z_range is not None else _z_range(df, var)
    if z_range is None or len(df) == 0:
        return pd.DataFrame(columns=_LINE_COLS)
    levels = contour_breaks(z_range, bins, binwidth, breaks)
    ux, uy, Z = grid_from_xyz(df["x"], df["y"], df[var])
    frames = []
    for li, lev in enumerate(levels):
        for k, line in enumerate(isolines(ux, uy, Z, float(lev)), start=1):
            frames.append(
                pd.DataFrame(
                    {"x": line[:, 0], "y": line[:, 1], "level": float(lev), "piece": k}
                ).assign(_li=li)
            )
    if not frames:
        return pd.DataFrame(columns=_LINE_COLS)
    out = pd.concat(frames, ignore_index=True)
    top = float(np.max(levels))
    out["nlevel"] = out["level"] / top if top else np.nan
    return out[_LINE_COLS]


def contour_bands(df, var, z_range, bins=None, binwidth=None, breaks=None) -> pd.DataFrame:
    """Filled-band rows (see :class:`StatContourFilled`) of column ``var``."""
    z_range = z_range if z_range is not None else _z_range(df, var)
    if z_range is None or len(df) == 0:
        return pd.DataFrame(columns=_BAND_COLS)
    levels = np.asarray(contour_breaks(z_range, bins, binwidth, breaks), dtype=float)
    if len(levels) < 2:
        return pd.DataFrame(columns=_BAND_COLS)
    labels = isoband_labels(levels.tolist())
    ux, uy, Z = grid_from_xyz(df["x"], df["y"], df[var])
    frames = []
    for li in range(len(levels) - 1):
        lo, hi = float(levels[li]), float(levels[li + 1])
        for k, ring in enumerate(isoband(ux, uy, Z, lo, hi), start=1):
            frames.append(
                pd.DataFrame(
                    {
                        "x": ring[:, 0],
                        "y": ring[:, 1],
                        "level_low": lo,
                        "level_high": hi,
                        "subgroup": k,
                        "_li": li,
                    }
                )
            )
    if not frames:
        return pd.DataFrame(columns=_BAND_COLS)
    out = pd.concat(frames, ignore_index=True)
    out["level"] = pd.Categorical.from_codes(out["_li"].to_numpy(), categories=labels, ordered=True)
    out["level_mid"] = (out["level_low"] + out["level_high"]) / 2
    top = float(np.max(levels[1:]))
    out["nlevel"] = out["level_high"] / top if top else np.nan
    out["piece"] = 1
    return out[_BAND_COLS]


def _stat_layer(stat: str, geom: Any, mapping, data, position, inherit_aes, name, params):
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
def stat_contour(
    mapping=None,
    data=None,
    *,
    geom="contour",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Contour lines of ``aes(x, y, z)`` (``bins``, ``binwidth`` or ``breaks``)."""
    return _stat_layer("contour", geom, mapping, data, position, inherit_aes, name, params)


@register
def stat_contour_filled(
    mapping=None,
    data=None,
    *,
    geom="contour_filled",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Filled contour bands of ``aes(x, y, z)``; ``fill=after_stat("level")``."""
    return _stat_layer("contour_filled", geom, mapping, data, position, inherit_aes, name, params)
