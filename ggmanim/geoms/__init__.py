"""Geoms: turn computed layer data into manim mobjects.

See :mod:`ggmanim.geoms.base` for the contract with the build pipeline and
legend guides. One public module per geom family; the export block below is
generated from each module's ``__all__`` by ``scripts/gen_exports.py``.
"""

# -- generated exports (scripts/gen_exports.py) --
from .annotate import annotate
from .base import KEY_SIZE, Geom, attach_marks, group_label, make_layer, resolve_color
from .col import GeomCol, geom_bar, geom_col, resolution
from .function import GeomFunction, geom_function
from .line import GeomLine, GeomPath, geom_line, geom_path
from .point import GeomPoint, geom_point
from .smooth import GeomSmooth, geom_smooth
from .text import GeomLabel, GeomText, geom_label, geom_text
from .vline import GeomHline, GeomVline, geom_hline, geom_vline

__all__ = [
    "Geom",
    "GeomCol",
    "GeomFunction",
    "GeomHline",
    "GeomLabel",
    "GeomLine",
    "GeomPath",
    "GeomPoint",
    "GeomSmooth",
    "GeomText",
    "GeomVline",
    "KEY_SIZE",
    "annotate",
    "attach_marks",
    "geom_bar",
    "geom_col",
    "geom_function",
    "geom_hline",
    "geom_label",
    "geom_line",
    "geom_path",
    "geom_point",
    "geom_smooth",
    "geom_text",
    "geom_vline",
    "group_label",
    "make_layer",
    "resolution",
    "resolve_color",
]
# -- end generated --
