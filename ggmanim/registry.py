"""Component registry.

Every public component constructor (``geom_*``, ``stat_*``, ``scale_*``,
``coord_*``, ``theme*``, ``labs``, ``xlab``, ``ylab``, ``xlim``, ``ylim``, ...)
is registered once with :func:`register`. :class:`ggmanim.plot.ggplot` looks
names up in :data:`COMPONENTS` *at call time* to provide chaining sugar, so
components defined in modules imported after ``plot.py`` still get a method::

    ggplot(df).geom_point(alpha=0.5)  ==  ggplot(df).add(geom_point(alpha=0.5))

A component constructor returns an object implementing
``add_to(self, plot) -> plot`` that returns a *new* plot.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, overload

__all__ = ["COMPONENTS", "register"]

F = TypeVar("F", bound=Callable)

#: name -> component constructor (function or class).
COMPONENTS: dict[str, Callable] = {}


@overload
def register(fn_or_cls: F) -> F: ...


@overload
def register(fn_or_cls: None = None, *, name: str | None = None) -> Callable[[F], F]: ...


def register(fn_or_cls: F | None = None, *, name: str | None = None) -> F | Callable[[F], F]:
    """Register a component constructor under its ``__name__`` (or ``name``).

    Usable as ``@register``, ``@register()`` or ``@register(name="alias")``.
    Returns the constructor unchanged. Re-registering a name replaces the
    previous entry (so module reloads work).
    """

    def _register(obj: F) -> F:
        key = name if name is not None else getattr(obj, "__name__", None)
        if not key or not isinstance(key, str):
            raise TypeError(f"cannot register {obj!r}: no name; pass register(name=...)")
        if key.startswith("_"):
            raise ValueError(f"component names must not start with '_': {key!r}")
        if not callable(obj):
            raise TypeError(f"component {key!r} is not callable")
        COMPONENTS[key] = obj
        return obj

    if fn_or_cls is None:
        return _register
    if isinstance(fn_or_cls, str):
        raise TypeError("use register(name=...) to register under an explicit name")
    return _register(fn_or_cls)
