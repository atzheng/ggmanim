"""geom_bin_2d: rectangular 2-D bin counts (``stat_bin_2d``) drawn as tiles."""

from __future__ import annotations

from ..registry import register
from .contour import _layer
from .rect import GeomTile

__all__ = ["GeomBin2d", "geom_bin_2d", "geom_bin2d"]


class GeomBin2d(GeomTile):
    """:class:`~ggmanim.geoms.rect.GeomTile` fed by ``stat_bin_2d`` (one tile per
    non-empty bin, ``fill=after_stat("count")``)."""

    default_stat = "bin_2d"

    def aesthetics(self) -> frozenset:
        return super().aesthetics() | {"weight"}


@register
def geom_bin_2d(
    mapping=None,
    data=None,
    *,
    stat="bin_2d",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Heatmap of 2-D bin counts of ``aes(x, y)``: ``bins`` (30), ``binwidth`` or
    ``breaks`` (one value or an ``(x, y)`` pair), ``drop`` empty bins (True).
    A ``weight`` aesthetic sums weights instead of counting."""
    return _layer(GeomBin2d(), stat, mapping, data, position, inherit_aes, name, params)


geom_bin2d = register(name="geom_bin2d")(geom_bin_2d)
