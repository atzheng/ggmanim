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
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import numpy as np
from manim import ManimColor

from ._viridis import VIRIDIS_TABLES

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
    "ContinuousPalette",
    "VIRIDIS_TABLES",
    "div_gradient_pal",
    "gradient_n_pal",
    "oklab_to_rgb",
    "rgb_to_oklab",
    "seq_gradient_pal",
    "viridis_pal",
]

Palette: TypeAlias = Sequence[Any] | Callable[[int], Sequence[Any]]


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
    [
        "#332288",
        "#117733",
        "#CC6677",
        "#88CCEE",
        "#999933",
        "#882255",
        "#44AA99",
        "#DDCC77",
        "#AA4499",
    ]
)
"""Paul Tol's "muted" qualitative scheme."""

VIBRANT: list[ManimColor] = _colors(
    ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#EE3377", "#BBBBBB"]
)
"""Paul Tol's "vibrant" qualitative scheme."""

SHAPES: list[str] = ["circle", "triangle", "square", "cross", "diamond", "star"]
"""Default palette for the ``shape`` aesthetic (names interpreted by geoms)."""

LINETYPES: list[str] = ["solid", "dashed", "dotted", "dashdot", "longdash", "twodash"]
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
        return (
            [ManimColor.from_rgb(colorsys.hsv_to_rgb(i / n, s, v)) for i in range(n)]
            if n > 0
            else []
        )

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


# ---------------------------------------------------------------------------
# Continuous (gradient) palettes: callables ``t in [0, 1] -> colours``
# ---------------------------------------------------------------------------

#: A continuous palette maps an array of positions in ``[0, 1]`` to a list of
#: colours (``None`` where the position is NaN).
ContinuousPalette: TypeAlias = Callable[[Any], list[Any]]


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


_RGB_TO_LMS = np.array(
    [
        [0.4122214708, 0.2119034982, 0.0883024619],
        [0.5363325363, 0.6806995451, 0.2817188376],
        [0.0514459929, 0.1073969566, 0.6299787005],
    ]
)
_LMS_TO_OKLAB = np.array(
    [
        [0.2104542553, 1.9779984951, 0.0259040371],
        [0.7936177850, -2.4285922050, 0.7827717662],
        [-0.0040720468, 0.4505937099, -0.8086757660],
    ]
)
_OKLAB_TO_LMS = np.array(
    [
        [1.0, 1.0, 1.0],
        [0.3963377774, -0.1055613458, -0.0894841775],
        [0.2158037573, -0.0638541728, -1.2914855480],
    ]
)
_LMS_TO_RGB = np.array(
    [
        [4.0767416621, -1.2684380046, -0.0041960863],
        [-3.3077115913, 2.6097574011, -0.7034186147],
        [0.2309699292, -0.3413193965, 1.7076147010],
    ]
)


def rgb_to_oklab(rgb: Any) -> np.ndarray:
    """sRGB (``(..., 3)`` floats in ``[0, 1]``) -> Oklab ``(L, a, b)`` (Ottosson, 2020)."""
    lin = _srgb_to_linear(np.asarray(rgb, dtype=float))
    return np.cbrt(lin @ _RGB_TO_LMS) @ _LMS_TO_OKLAB


def oklab_to_rgb(lab: Any) -> np.ndarray:
    """Oklab ``(..., 3)`` -> sRGB floats in ``[0, 1]`` (clipped to the gamut)."""
    lms = (np.asarray(lab, dtype=float) @ _OKLAB_TO_LMS) ** 3
    return _linear_to_srgb(lms @ _LMS_TO_RGB)


def _rgb_array(colours: Sequence[Any]) -> np.ndarray:
    return np.array([as_manim_color(c).to_rgb() for c in colours], dtype=float).reshape(-1, 3)


def gradient_n_pal(
    colours: Sequence[Any], values: Sequence[float] | None = None
) -> ContinuousPalette:
    """Continuous palette interpolating ``colours`` in Oklab space.

    ``values``: positions of the colours in ``[0, 1]`` (default: evenly spaced).
    Returns ``f(t) -> list[ManimColor | None]`` for an array ``t`` of positions in
    ``[0, 1]``; NaN positions give ``None``, positions outside are clamped.
    """
    cols = list(colours)
    if not cols:
        raise ValueError("gradient_n_pal() needs at least one colour")
    if values is None:
        pos = np.linspace(0.0, 1.0, len(cols)) if len(cols) > 1 else np.zeros(1)
    else:
        pos = np.asarray(values, dtype=float).ravel()
        if pos.size != len(cols):
            raise ValueError(f"got {pos.size} values for {len(cols)} colours")
        if np.any(np.diff(pos) < 0):
            raise ValueError("values must be increasing")
    lab = rgb_to_oklab(_rgb_array(cols))

    def pal(t: Any) -> list[Any]:
        x = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
        nan = ~np.isfinite(x)
        xc = np.where(nan, 0.0, x)
        interp = np.stack([np.interp(xc, pos, lab[:, k]) for k in range(3)], axis=-1)
        rgb = np.rint(oklab_to_rgb(interp) * 255).astype(int)
        return [
            None if bad else ManimColor("#{:02X}{:02X}{:02X}".format(*c))
            for c, bad in zip(rgb, nan)
        ]

    pal.__name__ = "gradient_n_pal"
    return pal


def seq_gradient_pal(low: Any = "#132B43", high: Any = "#56B1F7") -> ContinuousPalette:
    """Two-colour sequential gradient (defaults: ggplot2's default continuous colours)."""
    return gradient_n_pal([low, high])


def div_gradient_pal(
    low: Any = "#832424", mid: Any = "#FFFFFF", high: Any = "#3A3A98"
) -> ContinuousPalette:
    """Three-colour diverging gradient ``low -> mid -> high`` (mid at position 0.5)."""
    return gradient_n_pal([low, mid, high])


#: R-style single-letter option codes for :func:`viridis_pal`.
_VIRIDIS_OPTIONS = {"A": "magma", "B": "inferno", "C": "plasma", "D": "viridis", "E": "cividis"}


def _viridis_table(option: str) -> tuple[str, ...]:
    key = _VIRIDIS_OPTIONS.get(option, option).lower()
    try:
        return VIRIDIS_TABLES[key]
    except KeyError:
        raise ValueError(
            f"Unknown viridis option {option!r}; choose from {sorted(VIRIDIS_TABLES)}"
        ) from None


def viridis_pal(
    begin: float = 0.0, end: float = 1.0, direction: int = 1, option: str = "viridis"
) -> Callable[[int], list[ManimColor]]:
    """Discrete palette sampling a viridis-family map: ``n -> n colours``.

    ``option``: ``"viridis"`` (default), ``"magma"``, ``"inferno"``, ``"plasma"``,
    ``"cividis"`` (or R's letters ``"A"``..``"E"``). ``begin``/``end`` in ``[0, 1]``
    select the part of the map used; ``direction=-1`` reverses it.
    """
    table = _viridis_table(option)
    if not (0 <= begin <= 1 and 0 <= end <= 1):
        raise ValueError("begin and end must be in [0, 1]")
    if direction not in (1, -1):
        raise ValueError("direction must be 1 or -1")

    def pal(n: int) -> list[ManimColor]:
        if n <= 0:
            return []
        lo, hi = (begin, end) if direction == 1 else (end, begin)
        pos = np.linspace(lo, hi, n) if n > 1 else np.array([lo])
        idx = np.rint(pos * (len(table) - 1)).astype(int)
        return [ManimColor(table[i]) for i in idx]

    pal.__name__ = "viridis_pal"
    return pal


PALETTES.update(
    {
        name: viridis_pal(option=name)
        for name in ("viridis", "magma", "inferno", "plasma", "cividis")
    }
)
