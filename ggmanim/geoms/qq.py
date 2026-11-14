"""geom_qq / geom_qq_line: Q-Q plot points and reference line (stat_qq / stat_qq_line)."""

from __future__ import annotations

from ..registry import register
from .count import stat_layer
from .line import GeomPath
from .point import GeomPoint

__all__ = ["GeomQq", "GeomQqLine", "geom_qq", "geom_qq_line"]


class GeomQq(GeomPoint):
    """:class:`GeomPoint` fed by ``stat_qq`` (``x`` theoretical, ``y`` sample quantiles)."""

    default_stat = "qq"


class GeomQqLine(GeomPath):
    """:class:`GeomPath` fed by ``stat_qq_line``: one two-point line per group."""

    default_stat = "qq_line"


@register
def geom_qq(
    mapping=None,
    data=None,
    *,
    distribution=None,
    dparams=None,
    stat="qq",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Q-Q plot of ``aes(sample=...)`` against ``distribution`` (a quantile
    function ``q(p, **dparams)``; default the standard normal)."""
    params.update(distribution=distribution, dparams=dparams)
    return stat_layer(
        stat,
        GeomQq(),
        mapping,
        data,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )


@register
def geom_qq_line(
    mapping=None,
    data=None,
    *,
    distribution=None,
    dparams=None,
    line_p=(0.25, 0.75),
    fullrange: bool = False,
    stat="qq_line",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Reference line of a Q-Q plot through the ``line_p`` (25% / 75%) sample and
    theoretical quantiles; ``fullrange=True`` extends it over the x scale."""
    params.update(distribution=distribution, dparams=dparams, line_p=line_p, fullrange=fullrange)
    return stat_layer(
        stat,
        GeomQqLine(),
        mapping,
        data,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        params=params,
    )
