"""Scales: map data values to aesthetics; compute limits, breaks and labels.

No manim imports anywhere in this package.
"""

from .base import Scale
from .breaks import expand_range, expansion, format_numbers, log_breaks, pretty
from .continuous import (
    IdentityTransform,
    LogTransform,
    ReverseTransform,
    ScaleContinuous,
    ScaleContinuousPosition,
    SqrtTransform,
    Transform,
    censor,
    get_transform,
    scale_x_continuous,
    scale_x_log10,
    scale_x_reverse,
    scale_x_sqrt,
    scale_y_continuous,
    scale_y_log10,
    scale_y_reverse,
    scale_y_sqrt,
    squish,
    xlim,
    ylim,
)

__all__ = [
    "Scale",
    "ScaleContinuous",
    "ScaleContinuousPosition",
    "Transform",
    "IdentityTransform",
    "LogTransform",
    "SqrtTransform",
    "ReverseTransform",
    "get_transform",
    "censor",
    "squish",
    "pretty",
    "log_breaks",
    "format_numbers",
    "expansion",
    "expand_range",
    "scale_x_continuous",
    "scale_y_continuous",
    "scale_x_log10",
    "scale_y_log10",
    "scale_x_sqrt",
    "scale_y_sqrt",
    "scale_x_reverse",
    "scale_y_reverse",
    "xlim",
    "ylim",
]
