"""geom_histogram (stacked bars of ``stat_bin`` counts) and geom_freqpoly
(the same counts as a line through the bin centres)."""

from __future__ import annotations

from ..registry import register
from .col import GeomBar
from .count import stat_layer
from .line import GeomPath

__all__ = ["geom_histogram", "geom_freqpoly"]

_BIN_PARAMS = ("bins", "binwidth", "center", "boundary", "breaks", "closed", "pad")


def _bin_params(params: dict, **given) -> dict:
    """Add the binning params that differ from ``stat_bin``'s defaults (so a
    non-binning ``stat=`` gets no unknown-parameter warning for the defaults)."""
    defaults = {"closed": "right", "pad": False}
    params.update(
        {
            k: v
            for k, v in given.items()
            if k in _BIN_PARAMS and v is not None and not (k in defaults and v == defaults[k])
        }
    )
    return params


@register
def geom_histogram(
    mapping=None,
    data=None,
    *,
    stat="bin",
    position=None,
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
    """Histogram: bars of the counts of ``x`` in bins (``stat="bin"``), stacked by
    group. ``bins`` (default 30, with a warning) or ``binwidth``; ``center`` /
    ``boundary`` place the bins; ``breaks`` gives the edges. Map
    ``y=after_stat("density")`` (or ``"ncount"``/``"ndensity"``) for other scales;
    a ``weight`` aesthetic sums weights. See :class:`~ggmanim.stats.bin.StatBin`."""
    _bin_params(
        params,
        bins=bins,
        binwidth=binwidth,
        center=center,
        boundary=boundary,
        breaks=breaks,
        closed=closed,
        pad=pad,
    )
    return stat_layer(
        stat,
        GeomBar(),
        mapping,
        data,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )


@register
def geom_freqpoly(
    mapping=None,
    data=None,
    *,
    stat="bin",
    position="identity",
    bins: int | None = None,
    binwidth=None,
    center=None,
    boundary=None,
    breaks=None,
    closed: str = "right",
    pad: bool = True,
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Frequency polygon: a line through the bin counts of ``x``, one per group
    (``stat="bin"`` with ``pad=True``, so each line starts and ends at 0). Takes
    the same binning parameters as :func:`geom_histogram`."""
    _bin_params(
        params,
        bins=bins,
        binwidth=binwidth,
        center=center,
        boundary=boundary,
        breaks=breaks,
        closed=closed,
        pad=pad,
    )
    return stat_layer(
        stat,
        GeomPath(),
        mapping,
        data,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
