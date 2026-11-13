"""geom_function: draw ``y = fun(x)`` over the x scale's range (via stat_function)."""

from __future__ import annotations

from typing import Callable, Optional

from ..registry import register
from .base import make_layer
from .line import GeomPath

__all__ = ["GeomFunction", "geom_function"]


class GeomFunction(GeomPath):
    """A :class:`GeomPath` fed by ``stat_function`` (data already ordered by x).

    The layer has no data of its own before the stat, so it contributes nothing
    to pre-stat position training; after the stat its ``x``/``y`` train the scales
    like any line. ``label=`` is drawn at the curve end.
    """

    default_stat = "function"


@register
def geom_function(fun: Optional[Callable] = None, mapping=None, data=None, *, n: int = 101,
                  xlim=None, args=None, stat="function", position="identity",
                  inherit_aes: bool = False, name: Optional[str] = None, **params):
    """Draw the curve ``y = fun(x, **args)`` at ``n`` points.

    The x range is ``xlim`` if given, else the x scale's limits trained by the
    other layers. ``function=`` is accepted as a legacy spelling of ``fun``.
    Constant aesthetics (``color=``, ``linewidth=``) and ``label=`` go in ``params``.
    Like ggplot2, the layer does not inherit the plot's aesthetics.
    """
    legacy = params.pop("function", None)
    if fun is None:
        fun = legacy
    elif legacy is not None:
        raise TypeError("pass only one of fun= and function=")
    if fun is None or not callable(fun):
        raise TypeError("geom_function requires a callable `fun`")
    params.update(fun=fun, n=n, xlim=xlim, args=args)
    return make_layer(GeomFunction(), mapping, data, stat=stat, position=position,
                      inherit_aes=inherit_aes, name=name, params=params)
