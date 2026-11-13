"""Colour (and other discrete) palettes.

A *palette* is either

* a plain list of values (``ManimColor`` instances for colour palettes, or
  e.g. shape / linetype names), or
* a callable ``n -> list`` that generates ``n`` values on demand (e.g.
  :func:`hue_pal`).

Named palettes can also be referred to by string, see :data:`PALETTES` and
:func:`resolve_palette`.
"""

from __future__ import annotations

import colorsys
import math
from typing import Any, Callable, Sequence, Union

from manim import ManimColor

__all__ = [
    "BRIGHT",
    "MUTED",
    "VIBRANT",
    "SHAPES",
    "LINETYPES",
    "PALETTES",
    "Palette",
    "as_manim_color",
    "hcl_to_hex",
    "hue_pal",
    "hsv_pal",
    "resolve_palette",
]

Palette = Union[Sequence[Any], Callable[[int], Sequence[Any]]]


def as_manim_color(value: Any) -> ManimColor:
    """Coerce a colour-like value (hex string, name, RGB tuple, ManimColor) to ``ManimColor``."""
    if isinstance(value, ManimColor):
        return value
    return ManimColor(value)


def _colors(hexes: Sequence[str]) -> list[ManimColor]:
    return [ManimColor(h) for h in hexes]


# Paul Tol's colour-blind safe qualitative schemes
# https://personal.sron.nl/~pault/
BRIGHT: list[ManimColor] = _colors(
    ["#77AADD", "#BBCC33", "#EE8866", "#99DDFF", "#AAAA00", "#FFAABB", "#44BB99", "#EEDD88"]
)
"""Paul Tol's "light" qualitative scheme (historically called BRIGHT in ggmanim)."""

MUTED: list[ManimColor] = _colors(
    ["#332288", "#117733", "#CC6677", "#88CCEE", "#999933", "#882255", "#44AA99", "#DDCC77", "#AA4499"]
)
"""Paul Tol's "muted" qualitative scheme."""

VIBRANT: list[ManimColor] = _colors(
    ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#EE3377", "#BBBBBB"]
)
"""Paul Tol's "vibrant" qualitative scheme."""

SHAPES: list[str] = ["circle", "triangle", "square", "cross", "diamond", "star"]
"""Default palette for the ``shape`` aesthetic (names interpreted by geoms)."""

LINETYPES: list[str] = ["solid", "dashed", "dotted", "dashdot"]
"""Default palette for the ``linetype`` aesthetic (names interpreted by geoms)."""


# ---------------------------------------------------------------------------
# HCL hue palette (ggplot2's scales::hue_pal)
# ---------------------------------------------------------------------------

# D65 white point
_XN, _YN, _ZN = 0.95047, 1.0, 1.08883
_UN = 4 * _XN / (_XN + 15 * _YN + 3 * _ZN)
_VN = 9 * _YN / (_XN + 15 * _YN + 3 * _ZN)


def _gamma(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def hcl_to_hex(h: float, c: float, l: float) -> str:
    """Convert a polar-CIELUV (HCL) colour to an sRGB hex string (clipped to gamut)."""
    if l <= 0:
        return "#000000"
    hr = math.radians(h)
    u, v = c * math.cos(hr), c * math.sin(hr)
    y = _YN * (((l + 16) / 116) ** 3 if l > 8 else l / 903.2962962962963)
    up = u / (13 * l) + _UN
    vp = v / (13 * l) + _VN
    x = y * 9 * up / (4 * vp)
    z = y * (12 - 3 * up - 20 * vp) / (4 * vp)
    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    rgb = [min(max(_gamma(ch), 0.0), 1.0) for ch in (r, g, b)]
    return "#" + "".join(f"{round(ch * 255):02X}" for ch in rgb)


def hue_pal(
    h: tuple[float, float] = (15, 375),
    c: float = 100,
    l: float = 65,
    h_start: float = 0,
    direction: int = 1,
) -> Callable[[int], list[ManimColor]]:
    """ggplot2's default discrete palette: ``n`` evenly spaced hues in HCL space.

    Returns a callable ``n -> list[ManimColor]``.
    """

    def pal(n: int) -> list[ManimColor]:
        if n <= 0:
            return []
        h0, h1 = h
        # avoid a duplicated first/last hue when the range spans the full circle
        if (h1 - h0) % 360 < 1:
            h1 -= 360 / n
        if n == 1:
            hues = [h0]
        else:
            hues = [h0 + (h1 - h0) * i / (n - 1) for i in range(n)]
        hues = [(hh + h_start) % 360 for hh in hues]
        if direction == -1:
            hues = hues[::-1]
        return [ManimColor(hcl_to_hex(hh, c, l)) for hh in hues]

    pal.__name__ = "hue_pal"
    return pal


def hsv_pal(s: float = 0.65, v: float = 0.9) -> Callable[[int], list[ManimColor]]:
    """Simple evenly spaced HSV hues (cheap alternative to :func:`hue_pal`)."""

    def pal(n: int) -> list[ManimColor]:
        return [ManimColor.from_rgb(colorsys.hsv_to_rgb(i / n, s, v)) for i in range(n)] if n > 0 else []

    return pal


PALETTES: dict[str, Palette] = {
    "hue": hue_pal(),
    "bright": BRIGHT,
    "muted": MUTED,
    "vibrant": VIBRANT,
    "shapes": SHAPES,
    "linetypes": LINETYPES,
}


def resolve_palette(palette: Palette | str, n: int, *, cycle: bool = False) -> list[Any]:
    """Return exactly ``n`` values from ``palette``.

    ``palette`` may be a name in :data:`PALETTES`, a list, or a callable ``n -> list``.
    Raises ``ValueError`` if a fixed palette has fewer than ``n`` entries, unless
    ``cycle=True`` in which case entries are recycled.
    """
    if isinstance(palette, str):
        try:
            palette = PALETTES[palette.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown palette {palette!r}; available: {sorted(PALETTES)}"
            ) from None
    if callable(palette):
        values = list(palette(n))
    else:
        values = list(palette)
    if n == 0:
        return []
    if len(values) < n:
        if cycle and values:
            return [values[i % len(values)] for i in range(n)]
        raise ValueError(
            f"Palette has {len(values)} values but {n} levels need mapping. "
            "Use a larger palette, a callable palette such as hue_pal(), or cycle=True."
        )
    return values[:n]
