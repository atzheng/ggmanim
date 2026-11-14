"""stat_bin: counts of ``x`` in bins (for geom_histogram / geom_freqpoly).

The bin edges are a port of ggplot2's ``bin_breaks_width`` / ``bin_breaks_bins``
(``R/bin.R``): with neither ``center`` nor ``boundary``, ``bins`` picks the
width ``range / (bins - 1)`` so the data's extremes sit in the outer half of the
end bins, and ``binwidth`` puts a boundary at ``binwidth / 2`` (bins centred on
multiples of ``binwidth``). Intervals are closed on the right (``closed="left"``
flips that), the outermost edge is always included, and the edges are widened by
a fuzz of ``1e-8 * median(diff(breaks))`` so values on an edge are not lost to
rounding. Everything happens in transformed space: on ``scale_x_log10`` the bins
are equal in log units (as in ggplot2); ``breaks=`` are given in data units and
transformed by the x scale.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

from ..aes import after_stat
from ..registry import register
from .base import Stat, scale_limits
from .count import check_x_only, weights

__all__ = ["StatBin", "stat_bin", "bin_breaks", "bin_breaks_width", "bin_breaks_bins"]

#: ggplot2's message when neither ``bins`` nor ``binwidth``/``breaks`` is given.
BINS_WARNING = "stat_bin() using bins = 30. Pick better value with binwidth."


def _r_seq(start: float, stop: float, by: float) -> np.ndarray:
    """R's ``seq(start, stop, by)``."""
    n = math.floor((stop - start) / by + 1e-10)
    if n > 1e6:
        raise ValueError("the number of histogram bins must be less than 1,000,000")
    return start + by * np.arange(n + 1)


def bin_breaks_width(
    x_range, width: float, center: float | None = None, boundary: float | None = None
) -> np.ndarray:
    """Bin edges of width ``width`` covering ``x_range`` (ggplot2's ``bin_breaks_width``).

    ``boundary`` is the position of one edge, ``center`` of one bin centre (not
    both); with neither, ``boundary = width / 2``."""
    width = float(width)
    if not (width > 0 and math.isfinite(width)):
        raise ValueError("binwidth must be positive")
    if boundary is not None and center is not None:
        raise ValueError("only one of boundary and center may be specified")
    if boundary is None:
        boundary = width / 2 if center is None else center - width / 2
    lo, hi = float(x_range[0]), float(x_range[1])
    shift = math.floor((lo - boundary) / width)
    origin = boundary + shift * width
    # small correction so that e.g. origin = 0, max(x) = 20, width = 10 gives no extra bin
    max_x = hi + (1 - 1e-8) * width
    breaks = _r_seq(origin, max_x, width)
    if len(breaks) == 1:  # ggplot2 #3606: rare rounding case
        breaks = np.append(breaks, breaks[0] + width)
    return breaks


def bin_breaks_bins(
    x_range, bins: int = 30, center: float | None = None, boundary: float | None = None
) -> np.ndarray:
    """``bins`` bin edges covering ``x_range`` (ggplot2's ``bin_breaks_bins``)."""
    bins = int(bins)
    if bins < 1:
        raise ValueError("bins must be 1 or greater")
    lo, hi = float(x_range[0]), float(x_range[1])
    if hi - lo <= 1e-10 * max(1.0, abs(lo), abs(hi)):  # zero range
        width = 0.1
    elif bins == 1:
        width = hi - lo
        boundary = lo
        center = None
    else:
        width = (hi - lo) / (bins - 1)
        if center is None and boundary is None:
            boundary = lo - width / 2
        # if the range coincides with the boundary, use exactly `bins` bins
        if boundary is not None and (
            lo % width == boundary % width or hi % width == boundary % width
        ):
            width = (hi - lo) / bins
    return bin_breaks_width((lo, hi), width, center=center, boundary=boundary)


def bin_breaks(breaks, closed: str = "right") -> tuple[np.ndarray, np.ndarray]:
    """``(breaks, fuzzy)``: sorted edges and the fuzzed edges used for counting."""
    if closed not in ("right", "left"):
        raise ValueError(f"closed must be 'right' or 'left', got {closed!r}")
    breaks = np.sort(np.asarray(breaks, dtype=float))
    fin = breaks[np.isfinite(breaks)]
    fuzz = 1e-8 * float(np.median(np.diff(fin))) if len(fin) > 1 else np.nan
    if not math.isfinite(fuzz):
        fuzz = np.finfo(float).eps * 1e3
    fuzzes = np.full(len(breaks), fuzz)
    if closed == "right":
        fuzzes[0] = -fuzz
    else:
        fuzzes[:-1] = -fuzz
    return breaks, breaks + fuzzes


def bin_vector(
    x: np.ndarray, breaks: np.ndarray, closed: str, w: np.ndarray, pad: bool
) -> pd.DataFrame:
    """Counts of ``x`` (summed weights ``w``) per bin, ggplot2's ``bin_vector`` + ``bin_out``."""
    breaks, fuzzy = bin_breaks(breaks, closed)
    nb = len(breaks) - 1
    if closed == "right":  # (a, b], first bin closed
        idx = np.searchsorted(fuzzy, x, side="left") - 1
    else:  # [a, b), last bin closed
        idx = np.searchsorted(fuzzy, x, side="right") - 1
    ok = (idx >= 0) & (idx < nb) & np.isfinite(x)
    count = np.bincount(idx[ok], weights=w[ok], minlength=nb).astype(float)
    xc = (breaks[1:] + breaks[:-1]) / 2
    widths = np.diff(breaks)
    if pad:
        count = np.concatenate([[0.0], count, [0.0]])
        xc = np.concatenate([[xc[0] - widths[0]], xc, [xc[-1] + widths[-1]]])
        widths = np.concatenate([[widths[0]], widths, [widths[-1]]])
    total = float(np.abs(count).sum())
    with np.errstate(all="ignore"):
        density = count / widths / total if total else np.full(len(count), np.nan)
        cmax = float(np.abs(count).max()) if len(count) else 0.0
        dmax = float(np.nanmax(np.abs(density))) if total else np.nan
        return pd.DataFrame(
            {
                "count": count,
                "x": xc,
                "xmin": xc - widths / 2,
                "xmax": xc + widths / 2,
                "width": widths,
                "density": density,
                "ncount": count / cmax if cmax else np.full(len(count), np.nan),
                "ndensity": density / dmax if total and dmax else np.full(len(count), np.nan),
                "flipped_aes": False,
            }
        )


class StatBin(Stat):
    """Bins a continuous ``x`` (per group; edges shared by the whole panel).

    Parameters (``compute_group``): ``bins`` (default 30, with a warning; a
    number or a function of the x values), ``binwidth`` (likewise), ``center`` /
    ``boundary`` (one of), ``breaks`` (explicit edges in data units, or a
    function of the transformed x values returning them), ``closed``
    (``"right"``/``"left"``) and ``pad`` (an empty bin added each side).
    Binwidth, centre and boundary are in transformed units, as in ggplot2.

    Output columns, one row per bin: ``count`` (or summed ``weight``), ``x``
    (centre), ``xmin``, ``xmax``, ``width``, ``density`` (``count / width /
    total``), ``ncount``, ``ndensity`` (scaled to a maximum of 1),
    ``flipped_aes = False`` and ``y = count`` (overridden by the default
    ``y=after_stat("count")`` or any other ``after_stat`` mapping).
    """

    required_aes = ("x",)
    default_aes = {"y": after_stat("count")}
    dropped_aes = ("weight",)

    def compute_layer(self, df: pd.DataFrame, scales=None, **params) -> pd.DataFrame:
        if (
            params.get("bins") is None
            and params.get("binwidth") is None
            and params.get("breaks") is None
        ):
            warnings.warn(BINS_WARNING, UserWarning, stacklevel=3)
        return super().compute_layer(df, scales, **params)

    def compute_panel(self, df: pd.DataFrame, scales=None, **params) -> pd.DataFrame:
        check_x_only(self, df)
        xs = (scales or {}).get("x") if scales else None
        if getattr(xs, "is_discrete", False):
            raise ValueError(
                "stat_bin() requires a continuous x aesthetic; the x variable is discrete. "
                "Perhaps you want stat='count' (geom_bar())?"
            )
        if len(df):
            x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
            fin = x[np.isfinite(x)]
            if params.get("breaks") is not None:
                if callable(params["breaks"]):
                    params["breaks"] = params["breaks"](fin)
                elif xs is not None and hasattr(xs, "transform"):  # data units -> transformed
                    params["breaks"] = np.asarray(xs.transform(params["breaks"]), dtype=float)
            rng = scale_limits(scales, "x")
            if rng is None and len(fin):
                rng = (float(fin.min()), float(fin.max()))
            if rng is not None:
                params["_breaks"] = self._breaks(rng, fin, **params)
        return super().compute_panel(df, scales, **params)

    @staticmethod
    def _breaks(
        x_range,
        x: np.ndarray,
        bins=None,
        binwidth: float | Callable | None = None,
        center=None,
        boundary=None,
        breaks=None,
        **_,
    ) -> np.ndarray:
        if breaks is not None:  # (a function of x is resolved in compute_panel)
            return np.asarray(breaks, dtype=float)
        if binwidth is not None:
            bw = binwidth(x) if callable(binwidth) else binwidth
            return bin_breaks_width(x_range, float(bw), center=center, boundary=boundary)
        if callable(bins):
            bins = bins(x)
        return bin_breaks_bins(x_range, 30 if bins is None else bins, center, boundary)

    def compute_group(  # type: ignore[override]
        self,
        df: pd.DataFrame,
        scales=None,
        bins: int | None = None,
        binwidth: float | Callable | None = None,
        center: float | None = None,
        boundary: float | None = None,
        breaks=None,
        closed: str = "right",
        pad: bool = False,
        _breaks: np.ndarray | None = None,
    ) -> pd.DataFrame:
        x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
        if _breaks is None:  # called directly, not through compute_panel
            fin = x[np.isfinite(x)]
            if not len(fin):
                return pd.DataFrame()
            _breaks = self._breaks(
                (fin.min(), fin.max()), fin, bins, binwidth, center, boundary, breaks
            )
        out = bin_vector(x, _breaks, closed, weights(df), pad)
        out["y"] = out["count"]
        return out


@register
def stat_bin(
    mapping=None,
    data=None,
    *,
    geom="bar",
    position="stack",
    bins: int | None = None,
    binwidth=None,
    center=None,
    boundary=None,
    breaks=None,
    closed: str = "right",
    pad: bool = False,
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Histogram bars of binned ``x`` (as :func:`~ggmanim.geoms.histogram.geom_histogram`).
    ``y=after_stat("count")`` by default; ``"density"``, ``"ncount"``,
    ``"ndensity"`` too. See :class:`StatBin` for the parameters. Returns a layer."""
    from ..geoms.count import stat_layer  # geoms import manim: keep stats free of it

    params.update(
        bins=bins,
        binwidth=binwidth,
        center=center,
        boundary=boundary,
        breaks=breaks,
        closed=closed,
        pad=pad,
    )
    return stat_layer(
        "bin",
        geom,
        mapping,
        data,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
