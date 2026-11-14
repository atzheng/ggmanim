"""geom_density_2d / geom_density_2d_filled: contours of a 2-D kernel density
estimate (:mod:`ggmanim.stats.density2d`)."""

from __future__ import annotations

from ..registry import register
from .contour import GeomContour, GeomContourFilled, _layer

__all__ = [
    "GeomDensity2d",
    "GeomDensity2dFilled",
    "geom_density_2d",
    "geom_density_2d_filled",
    "geom_density2d",
    "geom_density2d_filled",
]


class GeomDensity2d(GeomContour):
    """Density contour lines (one path per contour piece)."""

    default_stat = "density_2d"


class GeomDensity2dFilled(GeomContourFilled):
    """Filled density bands (one polygon with holes per band)."""

    default_stat = "density_2d_filled"


@register
def geom_density_2d(
    mapping=None,
    data=None,
    *,
    stat="density_2d",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Contour lines of the 2-D kernel density of ``aes(x, y)``.

    ``n`` (grid size, 100), ``h`` (bandwidths), ``adjust``, ``contour_var``
    (``"density"``/``"ndensity"``/``"count"``) and ``bins``/``binwidth``/``breaks``
    for the levels; ``contour=False`` returns the density grid instead."""
    return _layer(GeomDensity2d(), stat, mapping, data, position, inherit_aes, name, params)


@register
def geom_density_2d_filled(
    mapping=None,
    data=None,
    *,
    stat="density_2d_filled",
    position="identity",
    inherit_aes: bool = True,
    name: str | None = None,
    **params,
):
    """Filled bands of the 2-D kernel density of ``aes(x, y)``, ``fill`` mapped to
    ``after_stat("level")``; same parameters as :func:`geom_density_2d`."""
    return _layer(GeomDensity2dFilled(), stat, mapping, data, position, inherit_aes, name, params)


geom_density2d = register(name="geom_density2d")(geom_density_2d)
geom_density2d_filled = register(name="geom_density2d_filled")(geom_density_2d_filled)
