"""Geoms: turn computed layer data into manim mobjects.

See :mod:`ggmanim.geoms.base` for the contract with the build pipeline and
legend guides.
"""

from .base import KEY_SIZE, Geom, make_layer
from .col import GeomCol, geom_bar, geom_col, resolution
from .function import GeomFunction, geom_function
from .line import GeomLine, GeomPath, geom_line, geom_path
from .point import GeomPoint, geom_point
from .smooth import GeomSmooth, geom_smooth
from .vline import GeomHline, GeomVline, geom_hline, geom_vline

__all__ = [
    "Geom",
    "GeomPoint",
    "GeomPath",
    "GeomLine",
    "GeomCol",
    "GeomVline",
    "GeomHline",
    "GeomFunction",
    "GeomSmooth",
    "KEY_SIZE",
    "make_layer",
    "resolution",
    "geom_point",
    "geom_path",
    "geom_line",
    "geom_col",
    "geom_bar",
    "geom_vline",
    "geom_hline",
    "geom_function",
    "geom_smooth",
]
