"""annotate(): a layer drawn from constant values instead of a data frame."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..aes import Aes, standardise_aes_name
from ..layer import Layer
from ..registry import COMPONENTS
from ..scales.continuous import X_AESTHETICS, Y_AESTHETICS
from .base import Geom

__all__ = ["annotate"]
__toplevel__ = ["annotate"]

#: Prefix of the data columns that carry vector-valued non-position aesthetics
#: past the non-position scales (so ``color=["red", "blue"]`` is used as is).
_RAW = "_annotate_"

#: Non-position aesthetics that have no default scale: vectors are mapped directly.
_UNSCALED = frozenset({"label", "angle", "hjust", "vjust", "family", "fontface"})


def _is_vector(v: Any) -> bool:
    return not isinstance(v, (str, bytes, Mapping)) and np.ndim(v) > 0


_GEOM_CLASSES: dict[type, type] = {}


def _annotation_geom_class(cls: type[Geom]) -> type[Geom]:
    """A subclass of ``cls`` (cached) whose :meth:`setup_data` first renames the
    ``_annotate_<aes>`` columns back to ``<aes>`` (raw, unscaled values)."""
    if cls in _GEOM_CLASSES:
        return _GEOM_CLASSES[cls]

    def setup_data(self, data: pd.DataFrame, params):
        raw = [c for c in data.columns if isinstance(c, str) and c.startswith(_RAW)]
        if raw:
            data = data.rename(columns={c: c[len(_RAW) :] for c in raw})
        return cls.setup_data(self, data, params)

    sub = type(cls.__name__, (cls,), {"setup_data": setup_data, "__module__": cls.__module__})
    _GEOM_CLASSES[cls] = sub
    return sub


def annotate(geom: str, x=None, y=None, *, name: str | None = None, **params) -> Layer:
    """A layer of ``geom`` (``"text"``, ``"label"``, ``"point"``, ``"rect"``,
    ``"segment"``, ...) drawn at constant positions, without a data frame.

    Position aesthetics (``x``, ``y``, ``xmin``, ``xmax``, ``ymin``, ``ymax``,
    ``xend``, ``yend``, ``xintercept``, ...) and ``label`` become the columns of a
    one-row data frame -- or an ``n``-row one when some are vectors of length
    ``n`` (scalars are recycled). Like ggplot2, the layer never inherits the
    plot's aesthetics, and it takes part in the axis limits. Vector values of other
    aesthetics (``color=["red", "blue"]``) are used per row *as is* (no scale,
    no legend); scalar ones and geom params (``size=``, ``parse=``, ``fill=``)
    are passed to the geom like constant aesthetics.

    The geom constructor is looked up as ``"geom_" + geom`` in
    :data:`ggmanim.registry.COMPONENTS`.

    >>> annotate("text", x=2, y=5, label="peak", color="red")  # doctest: +SKIP
    >>> annotate("rect", xmin=1, xmax=3, ymin=0, ymax=4, alpha=0.2)  # doctest: +SKIP
    """
    ctor = COMPONENTS.get(geom if geom.startswith("geom_") else "geom_" + geom)
    if ctor is None:
        known = sorted(k[5:] for k in COMPONENTS if k.startswith("geom_"))
        raise ValueError(f"annotate: unknown geom {geom!r}; available: {', '.join(known)}")
    params = {standardise_aes_name(k): v for k, v in params.items()}
    position_names = set(X_AESTHETICS) | set(Y_AESTHETICS)
    columns: dict[str, Any] = {}
    if x is not None:
        columns["x"] = x
    if y is not None:
        columns["y"] = y
    for k in list(params):
        if k in position_names or k == "label":
            columns[k] = params.pop(k)
    if not set(columns) & position_names:
        raise ValueError("annotate: give at least one position (x, y, xmin, xintercept, ...)")
    # vector-valued non-position aesthetics travel in the data as well
    vectors = {k: params.pop(k) for k in list(params) if _is_vector(params[k]) and k != "args"}
    lengths = {len(np.atleast_1d(np.asarray(v, dtype=object))) for v in columns.values()}
    lengths |= {len(v) for v in vectors.values()}
    n_set = lengths - {1}
    if len(n_set) > 1:
        raise ValueError(
            "annotate: unequal parameter lengths: "
            + ", ".join(
                f"{k} ({len(np.atleast_1d(np.asarray(v, dtype=object)))})"
                for k, v in {**columns, **vectors}.items()
            )
        )
    n = n_set.pop() if n_set else 1

    def column(v: Any) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(v, dtype=object) if not _is_vector(v) else list(v))
        out = np.empty(n, dtype=object)
        out[:] = list(arr) * (n // len(arr))
        return out

    frame: dict[str, Any] = {}
    mapping: dict[str, Any] = {}
    for k, v in columns.items():
        frame[k] = pd.Series(column(v)).infer_objects()
        mapping[k] = k
    raw_vectors = False
    for k, v in vectors.items():
        col = k if k in _UNSCALED else _RAW + k
        raw_vectors |= col != k
        frame[col] = pd.Series(column(v)).infer_objects()
        mapping[col] = col
    layer = ctor(Aes(mapping), pd.DataFrame(frame), **params)
    if not isinstance(layer, Layer):
        raise TypeError(f"annotate: geom_{geom} did not return a Layer")
    changes: dict[str, Any] = {"inherit_aes": False, "name": name}
    if raw_vectors:
        changes["geom"] = _annotation_geom_class(type(layer.geom))()
    return layer.replace(**changes)
