"""Facets: split a plot into panels (``facet_null``, ``facet_wrap``, ``facet_grid``).

Empty until the facet beads land. The export block below is generated from each
public module's ``__all__`` by ``scripts/gen_exports.py``.
"""

# -- generated exports (scripts/gen_exports.py) --
from .base import LAYOUT_COLUMNS, Facet, PanelGrid, PanelLayers
from .grid import MARGIN_LABEL, FacetGrid, facet_grid
from .labellers import as_labeller, label_both, label_value, labeller
from .null import FacetNull, facet_null
from .wrap import FacetWrap, facet_wrap, wrap_dims

__all__ = [
    "Facet",
    "FacetGrid",
    "FacetNull",
    "FacetWrap",
    "LAYOUT_COLUMNS",
    "MARGIN_LABEL",
    "PanelGrid",
    "PanelLayers",
    "as_labeller",
    "facet_grid",
    "facet_null",
    "facet_wrap",
    "label_both",
    "label_value",
    "labeller",
    "wrap_dims",
]
# -- end generated --
