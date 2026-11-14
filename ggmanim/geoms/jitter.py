"""geom_jitter: points with ``position_jitter`` (ggplot2's ``geom_jitter``)."""

from __future__ import annotations

from ..positions import PositionJitter
from ..registry import register
from .point import geom_point

__all__ = ["geom_jitter"]


@register
def geom_jitter(
    mapping=None,
    data=None,
    *,
    stat="identity",
    position="jitter",
    width: float | None = None,
    height: float | None = None,
    seed: int | None = None,
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Points with ``position_jitter(width, height, seed)`` (ggplot2's
    ``geom_jitter``): shorthand for ``geom_point(position=position_jitter(...))``.
    Giving both a ``position`` other than ``"jitter"`` and ``width``/``height``/
    ``seed`` is an error."""
    if position == "jitter" and (width is not None or height is not None or seed is not None):
        position = PositionJitter(width=width, height=height, seed=seed)
    elif width is not None or height is not None or seed is not None:
        raise ValueError("specify either `position` or `width`/`height`/`seed`, not both")
    return geom_point(
        mapping,
        data,
        stat=stat,
        position=position,
        inherit_aes=inherit_aes,
        name=name,
        **params,
    )
