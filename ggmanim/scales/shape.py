"""Shape and linetype scales, and the vocabulary of shape / linetype values.

No manim imports: the drawing lives in :mod:`ggmanim.geoms._markers` and
:mod:`ggmanim.geoms._linetype`, which use :func:`parse_shape` /
:func:`parse_linetype` from here.

Shapes
------
``circle``, ``triangle``, ``square``, ``diamond``, ``star`` (filled, with a
``_open`` hollow variant such as ``"circle_open"``) and the stroke-only
``cross`` (x) and ``plus`` (+). ggplot2's integer ``pch`` codes 0-25 are
accepted too (``16`` = solid circle, ``1`` = open circle, ...).

Linetypes
---------
``solid``, ``dashed``, ``dotted``, ``dashdot`` (alias ``dotdash``),
``longdash``, ``twodash``, ``blank``, ggplot2's integer codes 0-6, or a custom
on/off pattern given as a hex string (``"44"``, ``"1343"``) or a sequence of
numbers, in units of the line width (as in ggplot2).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..palettes import LINETYPES, SHAPES
from ..registry import register
from .discrete import ScaleDiscrete, ScaleDiscreteIdentity, ScaleDiscreteManual

__all__ = [
    "LINETYPE_PATTERNS",
    "SHAPE_NAMES",
    "SHAPES_OPEN",
    "parse_linetype",
    "parse_shape",
    "scale_linetype",
    "scale_linetype_identity",
    "scale_linetype_manual",
    "scale_shape",
    "scale_shape_identity",
    "scale_shape_manual",
]

_FILLABLE = ("circle", "triangle", "square", "diamond", "star")
_STROKE_ONLY = ("cross", "plus")

#: Every accepted shape name.
SHAPE_NAMES: tuple[str, ...] = (
    _FILLABLE + tuple(f"{s}_open" for s in _FILLABLE) + tuple(f"{s}_filled" for s in _FILLABLE)
) + _STROKE_ONLY

#: Default palette of ``scale_shape(solid=False)``: hollow variants of :data:`SHAPES`.
SHAPES_OPEN: list[str] = [s if s in _STROKE_ONLY else f"{s}_open" for s in SHAPES]

_SHAPE_ALIASES = {
    "dot": "circle",
    "point": "circle",
    "x": "cross",
    "times": "cross",
    "+": "plus",
    "o": "circle_open",
    "triangle_up": "triangle",
}

# ggplot2's pch codes -> shape names (closest available glyph).
_PCH = {
    0: "square_open",
    1: "circle_open",
    2: "triangle_open",
    3: "plus",
    4: "cross",
    5: "diamond_open",
    6: "triangle_open",
    7: "square_open",
    8: "star",
    9: "diamond_open",
    10: "circle_open",
    11: "star_open",
    12: "square_open",
    13: "circle_open",
    14: "square_open",
    15: "square",
    16: "circle",
    17: "triangle",
    18: "diamond",
    19: "circle",
    20: "circle",
    21: "circle_filled",
    22: "square_filled",
    23: "diamond_filled",
    24: "triangle_filled",
    25: "triangle_filled",
}


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def parse_shape(shape: Any) -> tuple[str, str]:
    """``shape`` -> ``(base, style)`` with ``style`` one of ``"solid"`` (filled
    with ``fill``, falling back to ``color``), ``"open"`` (outline in ``color``),
    ``"filled"`` (outline in ``color``, fill in ``fill``) or ``"stroke"``
    (cross / plus). ``None``/NaN means ``"circle"``.

    Raises ``ValueError`` listing the valid names for anything else.
    """
    if _is_missing(shape):
        return "circle", "solid"
    if isinstance(shape, bool):
        raise _shape_error(shape)
    if isinstance(shape, (int, float)) and float(shape).is_integer() and int(shape) in _PCH:
        shape = _PCH[int(shape)]
    if not isinstance(shape, str):
        raise _shape_error(shape)
    name = shape.strip().lower().replace(" ", "_").replace("-", "_")
    name = _SHAPE_ALIASES.get(name, name)
    if name in _STROKE_ONLY:
        return name, "stroke"
    if name in _FILLABLE:
        return name, "solid"
    for suffix in ("open", "filled"):
        if name.endswith("_" + suffix) and name[: -len(suffix) - 1] in _FILLABLE:
            return name[: -len(suffix) - 1], suffix
    raise _shape_error(shape)


def _shape_error(shape: Any) -> ValueError:
    return ValueError(
        f"Unknown shape {shape!r}. Valid shapes: {', '.join(SHAPE_NAMES)} "
        f"(default palette SHAPES = {SHAPES}), or a ggplot2 pch code 0-25."
    )


#: Dash patterns (alternating on/off lengths, in units of the line width);
#: ``None`` = solid, ``()`` = blank.
LINETYPE_PATTERNS: dict[str, tuple[float, ...] | None] = {
    "solid": None,
    "dashed": (4, 4),
    "dotted": (1, 3),
    "dashdot": (1, 3, 4, 3),
    "longdash": (7, 3),
    "twodash": (2, 2, 6, 2),
    "blank": (),
}

_LINETYPE_ALIASES = {"dotdash": "dashdot", "dash": "dashed", "dot": "dotted"}
_LTY_CODES = ["blank", "solid", "dashed", "dotted", "dashdot", "longdash", "twodash"]


def parse_linetype(linetype: Any) -> tuple[float, ...] | None:
    """``linetype`` -> on/off pattern in line-width units (see
    :data:`LINETYPE_PATTERNS`); ``None``/NaN (or ``"solid"``) -> ``None``.

    Raises ``ValueError`` listing the valid names for anything else.
    """
    if _is_missing(linetype):
        return None
    if isinstance(linetype, bool):
        raise _linetype_error(linetype)
    if isinstance(linetype, (int, float)):
        if float(linetype).is_integer() and 0 <= int(linetype) < len(_LTY_CODES):
            return LINETYPE_PATTERNS[_LTY_CODES[int(linetype)]]
        raise _linetype_error(linetype)
    if isinstance(linetype, str):
        name = linetype.strip().lower()
        name = _LINETYPE_ALIASES.get(name, name)
        if name in LINETYPE_PATTERNS:
            return LINETYPE_PATTERNS[name]
        hexdigits = "0123456789abcdef"
        if len(name) in (2, 4, 6, 8) and all(c in hexdigits for c in name):
            pattern = tuple(float(int(c, 16)) for c in name)
            if all(p > 0 for p in pattern):
                return pattern
        raise _linetype_error(linetype)
    if isinstance(linetype, Sequence):
        try:
            pattern = tuple(float(p) for p in linetype)
        except (TypeError, ValueError):
            raise _linetype_error(linetype) from None
        if len(pattern) >= 2 and len(pattern) % 2 == 0 and all(p > 0 for p in pattern):
            return pattern
    raise _linetype_error(linetype)


def _linetype_error(linetype: Any) -> ValueError:
    return ValueError(
        f"Unknown linetype {linetype!r}. Valid linetypes: {', '.join(LINETYPE_PATTERNS)} "
        f"(default palette LINETYPES = {LINETYPES}), an integer code 0-6, a hex "
        "dash string such as '44' or '1343', or an even-length sequence of positive "
        "on/off lengths (in line widths)."
    )


def _validate_values(values: Mapping[Any, Any] | Sequence[Any], parse) -> None:
    for v in values.values() if isinstance(values, Mapping) else values:
        parse(v)


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


@register
def scale_shape(solid: bool = True, *, aesthetics="shape", **kwargs) -> ScaleDiscrete:
    """Discrete shape scale: palette :data:`~ggmanim.palettes.SHAPES` (filled) or,
    with ``solid=False``, their hollow ``_open`` variants."""
    return ScaleDiscrete(aesthetics, palette=list(SHAPES if solid else SHAPES_OPEN), **kwargs)


@register
def scale_shape_manual(values, *, aesthetics="shape", **kwargs) -> ScaleDiscreteManual:
    """Shape scale with explicit ``values`` (dict level -> shape, or list in
    level order). Unknown shape names raise ``ValueError`` listing the valid ones."""
    _validate_values(values, parse_shape)
    return ScaleDiscreteManual(aesthetics, values=values, **kwargs)


@register
def scale_shape_identity(*, aesthetics="shape", **kwargs) -> ScaleDiscreteIdentity:
    """Use data values directly as shapes. No legend by default."""
    return ScaleDiscreteIdentity(aesthetics, **kwargs)


@register
def scale_linetype(*, aesthetics="linetype", **kwargs) -> ScaleDiscrete:
    """Discrete linetype scale (palette :data:`~ggmanim.palettes.LINETYPES`)."""
    return ScaleDiscrete(aesthetics, palette=list(LINETYPES), **kwargs)


@register
def scale_linetype_manual(values, *, aesthetics="linetype", **kwargs) -> ScaleDiscreteManual:
    """Linetype scale with explicit ``values`` (dict level -> linetype, or list in
    level order). Unknown linetypes raise ``ValueError`` listing the valid ones."""
    _validate_values(values, parse_linetype)
    return ScaleDiscreteManual(aesthetics, values=values, **kwargs)


@register
def scale_linetype_identity(*, aesthetics="linetype", **kwargs) -> ScaleDiscreteIdentity:
    """Use data values directly as linetypes. No legend by default."""
    return ScaleDiscreteIdentity(aesthetics, **kwargs)
