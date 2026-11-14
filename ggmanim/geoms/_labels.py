"""Shared text machinery for :mod:`.text` (``geom_text``/``geom_label``) and for
the *direct label* option of the line-like geoms.

Text size
---------
ggplot2 measures text ``size`` in millimetres with a default of ``3.88`` (11 pt).
Here ``size`` keeps that scale but is relative to the theme: a label of ``size``
is drawn at ``theme.axis_text_size * size / 3.88`` font units (see
:func:`font_size_for`), so ``size=3.88`` matches the tick labels and the text
scales with the theme.

Justification
-------------
:func:`place_text` puts the point ``(hjust, vjust)`` of the text's *unrotated*
bounding box (``0`` = left/bottom, ``1`` = right/top) on the anchor, then
rotates the text by ``angle`` degrees about the anchor -- ggplot2's semantics.
``hjust``/``vjust`` may also be ``"left"``, ``"center"``/``"middle"``,
``"right"``, ``"bottom"``, ``"top"``, ``"inward"`` or ``"outward"`` (see
:func:`resolve_just`).

Direct labels
-------------
``geom_line``/``geom_path``/``geom_function``/``geom_smooth``/``geom_vline``/
``geom_hline`` accept ``label=`` (a str or Mobject for every group/line, a
sequence in group order or a dict group id -> label). The label is drawn with
the same machinery as ``geom_text`` (:func:`direct_label`): in the group's
colour at ``theme.axis_text_size``, justified against an anchor -- right of a
line's last visible point (``hjust=0``), along the top-left of a vline
(rotated 90 degrees) and above the right end of an hline.

*Group order* is the order of the integer ``group`` ids, which
:func:`~ggmanim.stats.add_group` numbers ``1..k`` in **sorted** order of the
discrete grouping aesthetics' values (``color``/``fill``/``linetype``/... or an
explicit ``group``; category order for a pandas ``Categorical``), not in order
of appearance in the data. So with ``aes(color="species")`` and species
``"b", "a"`` in that order in the data, ``label=["A", "B"]`` puts ``"A"`` on
species ``"a"``. A dict is keyed by the raw grouping value (``{"a": "A"}``; a
tuple when several aesthetics group) or by those group ids (``{1: "A", 2: "B"}``;
``-1`` when there is no grouping aesthetic); the build re-keys raw values by
group id (:func:`~ggmanim.build.draw_layers`), and a raw value wins over an equal
group id. Groups missing from the dict, or beyond the end of a list, get no label. For
``geom_vline``/``geom_hline`` a list follows the order of the intercepts as
given (drawn rows) and a dict is keyed by that 0-based position.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, cast

import manim
import numpy as np
from manim import DEGREES, OUT, Mobject, VMobject

from ..theme import Theme, hjust_align

__all__ = [
    "DEFAULT_TEXT_SIZE",
    "bbox",
    "boxes_overlap",
    "direct_label",
    "font_size_for",
    "format_label",
    "place_text",
    "resolve_just",
    "text_mobject",
]

#: ggplot2's default text size (mm); maps to ``theme.axis_text_size``.
DEFAULT_TEXT_SIZE = 3.88

_HJUST = {"left": 0.0, "center": 0.5, "centre": 0.5, "middle": 0.5, "right": 1.0}
_VJUST = {"bottom": 0.0, "center": 0.5, "centre": 0.5, "middle": 0.5, "top": 1.0}


def font_size_for(theme: Theme, size: Any) -> float:
    """Font size for a ggplot2-style text ``size`` (``3.88`` -> ``axis_text_size``)."""
    try:
        s = float(size)
    except (TypeError, ValueError):
        s = DEFAULT_TEXT_SIZE
    if not np.isfinite(s) or s <= 0:
        s = DEFAULT_TEXT_SIZE
    return theme.axis_text_size * s / DEFAULT_TEXT_SIZE


def resolve_just(value: Any, pos: float | None = None, vertical: bool = False) -> float:
    """A justification as a number in ``[0, 1]``-ish (values outside are allowed).

    ``pos`` is the anchor's relative position in the panel (``0`` .. ``1``) used
    by ``"inward"`` (towards the panel centre) and ``"outward"`` (away from it);
    without it they mean ``0.5``.
    """
    if isinstance(value, str):
        v = value.lower()
        if v in ("inward", "outward"):
            if pos is None or abs(pos - 0.5) < 1e-9:
                return 0.5
            left = pos < 0.5
            return (0.0 if left else 1.0) if v == "inward" else (1.0 if left else 0.0)
        table = _VJUST if vertical else _HJUST
        if v not in table:
            raise ValueError(f"unknown {'vjust' if vertical else 'hjust'} {value!r}")
        return table[v]
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.5
    return f if np.isfinite(f) else 0.5


def format_label(value: Any) -> str | None:
    """A label value as text: ``None``/NaN -> ``None``; whole floats without
    ``.0`` (counts); other floats with up to 6 significant digits."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.6g}"
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def text_mobject(
    theme: Theme,
    label: Any,
    *,
    font_size: float | None = None,
    color: Any = None,
    alpha: float = 1.0,
    family: str | None = None,
    fontface: Any = None,
    parse: bool = False,
    hjust: float = 0.5,
) -> Mobject | None:
    """The mobject for one label, or ``None`` for an empty/missing label.

    Strings go through :meth:`Theme.make_text` (multi-line text is justified by
    ``hjust``). ``parse=True`` renders ``label`` as LaTeX math (``MathTex``) when
    ``theme.text_engine == "tex"``; with the Pango engine it warns and draws the
    plain text. ``family`` / ``fontface`` (``"plain"``, ``"bold"``, ``"italic"``,
    ``"bold.italic"``) apply to the Pango engine only. A Mobject is copied and
    recoloured.
    """
    if isinstance(label, Mobject):
        mob = label.copy()
        if color is not None:
            mob.set_color(color)
    else:
        text = format_label(label)
        if text is None or text == "":
            return None
        if font_size is None:
            font_size = theme.axis_text_size
        if color is None:
            color = theme.get_foreground()
        if parse and theme.text_engine == "tex":
            mob = manim.MathTex(text, font_size=font_size, color=color)
        else:
            if parse:
                warnings.warn(
                    "parse=True needs theme(text_engine='tex'); drawing the label as plain text",
                    stacklevel=3,
                )
            kwargs: dict[str, Any] = {}
            if theme.text_engine != "tex":
                if family:
                    kwargs["font"] = family
                face = str(fontface or "plain").lower()
                if "bold" in face:
                    kwargs["weight"] = manim.BOLD
                if "italic" in face:
                    kwargs["slant"] = manim.ITALIC
            mob = theme.make_text(
                text, font_size=font_size, color=color, align=hjust_align(hjust), **kwargs
            )
    if alpha is not None and alpha < 1.0:
        mob.set_opacity(alpha)
    return mob


def place_text(
    mob: Mobject,
    point: Sequence[float] | np.ndarray,
    hjust: float = 0.5,
    vjust: float = 0.5,
    angle: float = 0.0,
) -> Mobject:
    """Move ``mob`` so the ``(hjust, vjust)`` point of its unrotated bounding box
    sits on ``point``, rotated by ``angle`` degrees about ``point``. Returns ``mob``."""
    point = np.asarray(point, dtype=float)
    left, bottom = mob.get_left()[0], mob.get_bottom()[1]
    anchor = np.array([left + hjust * mob.width, bottom + vjust * mob.height, 0.0])
    mob.shift(point - anchor)
    if angle:
        mob.rotate(float(angle) * DEGREES, axis=OUT, about_point=point)
    return mob


def direct_label(
    theme: Theme,
    label: Any,
    color: Any,
    point: Sequence[float] | np.ndarray,
    *,
    hjust: float = 0.0,
    vjust: float = 0.5,
    angle: float = 0.0,
    nudge: Sequence[float] = (0.0, 0.0),
) -> VMobject | None:
    """A direct label (the ``label=`` option of line-like geoms): text in
    ``color`` at ``theme.axis_text_size`` justified against ``point`` shifted by
    ``nudge`` (scene units). ``None`` when ``label`` is empty."""
    mob = text_mobject(theme, label, color=color, hjust=hjust)
    if mob is None:
        return None
    anchor = np.asarray(point, dtype=float) + np.array([nudge[0], nudge[1], 0.0])
    return cast(VMobject, place_text(mob, anchor, hjust, vjust, angle))


def boxes_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two ``(xmin, ymin, xmax, ymax)`` boxes overlap (touching is not)."""
    return bool(a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])


def bbox(mob: Mobject) -> np.ndarray:
    """``(xmin, ymin, xmax, ymax)`` of ``mob`` in scene units."""
    lo, hi = mob.get_corner(manim.DL), mob.get_corner(manim.UR)
    return np.array([lo[0], lo[1], hi[0], hi[1]])
