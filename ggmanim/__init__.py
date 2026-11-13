"""ggmanim: a ggplot2-style grammar of graphics for manim.

Importing this package registers every component (``geom_*``, ``stat_*``,
``scale_*``, ``coord_*``, ``theme_*``, ...) so that ``ggplot(...).geom_point()``
style chaining works. The public API is exactly :data:`__all__`.
"""

from . import build as _build  # noqa: F401  (the pipeline; registers guides)
from .aes import aes, after_stat
from .coords import coord_cartesian
from .geoms import (
    geom_bar,
    geom_col,
    geom_function,
    geom_hline,
    geom_line,
    geom_point,
    geom_smooth,
    geom_vline,
)
from .guides import guides
from .palettes import (
    BRIGHT,
    LINETYPES,
    MUTED,
    PALETTES,
    SHAPES,
    VIBRANT,
    Palette,
    as_manim_color,
    hcl_to_hex,
    hsv_pal,
    hue_pal,
    resolve_palette,
)
from .plot import GGManim, ggplot, labs, xlab, ylab
from .scales import (
    scale_x_continuous,
    scale_x_log10,
    scale_x_reverse,
    scale_x_sqrt,
    scale_y_continuous,
    scale_y_log10,
    scale_y_reverse,
    scale_y_sqrt,
    xlim,
    ylim,
)
from .scales.discrete import (
    scale_color_discrete,
    scale_color_identity,
    scale_color_manual,
    scale_colour_discrete,
    scale_colour_identity,
    scale_colour_manual,
    scale_fill_discrete,
    scale_fill_identity,
    scale_fill_manual,
)
from .scales.discrete_position import scale_x_discrete, scale_y_discrete
from .stats import stat_function, stat_identity, stat_smooth
from .theme import theme, theme_classic, theme_default, theme_minimal

__version__ = "0.1.0"

__all__ = [
    # plot
    "ggplot",
    "GGManim",
    "aes",
    "after_stat",
    "labs",
    "xlab",
    "ylab",
    "guides",
    # geoms
    "geom_bar",
    "geom_col",
    "geom_function",
    "geom_hline",
    "geom_line",
    "geom_point",
    "geom_smooth",
    "geom_vline",
    # stats
    "stat_function",
    "stat_identity",
    "stat_smooth",
    # scales
    "scale_x_continuous",
    "scale_x_log10",
    "scale_x_reverse",
    "scale_x_sqrt",
    "scale_y_continuous",
    "scale_y_log10",
    "scale_y_reverse",
    "scale_y_sqrt",
    "scale_x_discrete",
    "scale_y_discrete",
    "scale_color_discrete",
    "scale_color_identity",
    "scale_color_manual",
    "scale_colour_discrete",
    "scale_colour_identity",
    "scale_colour_manual",
    "scale_fill_discrete",
    "scale_fill_identity",
    "scale_fill_manual",
    "xlim",
    "ylim",
    # coords
    "coord_cartesian",
    # themes
    "theme",
    "theme_classic",
    "theme_default",
    "theme_minimal",
    # palettes
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
