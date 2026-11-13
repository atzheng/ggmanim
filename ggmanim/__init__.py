# Legacy prototype API (removed in ggm-s11.10); the new API below wins on name clashes.
from .GGManim import *  # noqa: F401,F403
from .palettes import *  # noqa: F401,F403

# New grammar-of-graphics API. Importing these modules registers every component.
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
from .stats import stat_function, stat_identity, stat_smooth
from .theme import theme, theme_classic, theme_default, theme_minimal
from . import build as _build  # noqa: F401  (the pipeline; registers guides)
