"""Geoms: turn computed layer data into manim mobjects.

See :mod:`ggmanim.geoms.base` for the contract with the build pipeline and
legend guides. One public module per geom family; the export block below is
generated from each module's ``__all__`` by ``scripts/gen_exports.py``.
"""

# -- generated exports (scripts/gen_exports.py) --
from .abline import GeomAbline, geom_abline
from .annotate import annotate
from .area import GeomArea, GeomRibbon, geom_area, geom_ribbon
from .base import KEY_SIZE, Geom, attach_marks, group_label, make_layer, resolve_color
from .bin2d import GeomBin2d, geom_bin2d, geom_bin_2d
from .col import GeomBar, GeomCol, geom_bar, geom_col, resolution
from .contour import (
    GeomContour,
    GeomContourFilled,
    draw_oriented_polygons,
    geom_contour,
    geom_contour_filled,
)
from .count import GeomCount, geom_count
from .density2d import (
    GeomDensity2d,
    GeomDensity2dFilled,
    geom_density2d,
    geom_density2d_filled,
    geom_density_2d,
    geom_density_2d_filled,
)
from .function import GeomFunction, geom_function
from .hex import GeomHex, geom_hex
from .histogram import geom_freqpoly, geom_histogram
from .interval import (
    GeomCrossbar,
    GeomErrorbar,
    GeomErrorbarh,
    GeomLinerange,
    GeomPointrange,
    draw_crossbar,
    flip_data,
    geom_crossbar,
    geom_errorbar,
    geom_errorbarh,
    geom_linerange,
    geom_pointrange,
)
from .line import GeomLine, GeomPath, geom_line, geom_path
from .point import GeomPoint, geom_point
from .polygon import GeomPolygon, geom_polygon
from .qq import GeomQq, GeomQqLine, geom_qq, geom_qq_line
from .rect import GeomRaster, GeomRect, GeomTile, geom_raster, geom_rect, geom_tile
from .segment import ArrowSpec, GeomCurve, GeomSegment, arrow, geom_curve, geom_segment
from .smooth import GeomSmooth, geom_smooth
from .step import GeomStep, geom_step, stairstep
from .text import GeomLabel, GeomText, geom_label, geom_text
from .vline import GeomHline, GeomVline, geom_hline, geom_vline

__all__ = [
    "ArrowSpec",
    "Geom",
    "GeomAbline",
    "GeomArea",
    "GeomBar",
    "GeomBin2d",
    "GeomCol",
    "GeomContour",
    "GeomContourFilled",
    "GeomCount",
    "GeomCrossbar",
    "GeomCurve",
    "GeomDensity2d",
    "GeomDensity2dFilled",
    "GeomErrorbar",
    "GeomErrorbarh",
    "GeomFunction",
    "GeomHex",
    "GeomHline",
    "GeomLabel",
    "GeomLine",
    "GeomLinerange",
    "GeomPath",
    "GeomPoint",
    "GeomPointrange",
    "GeomPolygon",
    "GeomQq",
    "GeomQqLine",
    "GeomRaster",
    "GeomRect",
    "GeomRibbon",
    "GeomSegment",
    "GeomSmooth",
    "GeomStep",
    "GeomText",
    "GeomTile",
    "GeomVline",
    "KEY_SIZE",
    "annotate",
    "arrow",
    "attach_marks",
    "draw_crossbar",
    "draw_oriented_polygons",
    "flip_data",
    "geom_abline",
    "geom_area",
    "geom_bar",
    "geom_bin2d",
    "geom_bin_2d",
    "geom_col",
    "geom_contour",
    "geom_contour_filled",
    "geom_count",
    "geom_crossbar",
    "geom_curve",
    "geom_density2d",
    "geom_density2d_filled",
    "geom_density_2d",
    "geom_density_2d_filled",
    "geom_errorbar",
    "geom_errorbarh",
    "geom_freqpoly",
    "geom_function",
    "geom_hex",
    "geom_histogram",
    "geom_hline",
    "geom_label",
    "geom_line",
    "geom_linerange",
    "geom_path",
    "geom_point",
    "geom_pointrange",
    "geom_polygon",
    "geom_qq",
    "geom_qq_line",
    "geom_raster",
    "geom_rect",
    "geom_ribbon",
    "geom_segment",
    "geom_smooth",
    "geom_step",
    "geom_text",
    "geom_tile",
    "geom_vline",
    "group_label",
    "make_layer",
    "resolution",
    "resolve_color",
    "stairstep",
]
# -- end generated --
