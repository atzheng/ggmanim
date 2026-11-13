"""Legend guide options (:func:`guide_legend`) and the legend frame.

:func:`guide_legend` configures a key-glyph legend, either on the scale
(``scale_color_discrete(guide=guide_legend(ncol=2))``) or via
``guides(color=guide_legend(reverse=True))``. The legend itself
(:class:`ggmanim.guides.Legend`) and the merging of legends live in
:mod:`ggmanim.guides`.

:func:`legend_frame` builds the optional background / border rectangle drawn
behind a legend or colourbar (``theme(legend_background=..., legend_border=True)``);
:func:`place_legend_title` puts a legend's or colourbar's title beside its body
(``theme(legend_title_position=..., legend_title_hjust=...)``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from manim import DOWN, LEFT, RIGHT, UP, Mobject, SurroundingRectangle

from .aes import standardise_aes_name
from .theme import Theme

__all__ = [
    "GuideLegend",
    "guide_legend",
    "is_legend_spec",
    "legend_frame",
    "place_legend_title",
]

_DIRECTIONS = (None, "horizontal", "vertical")


def _pos_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer or None, got {value!r}")
    return int(value)


class GuideLegend:
    """Legend options (see :func:`guide_legend`)."""

    __slots__ = (
        "title",
        "nrow",
        "ncol",
        "byrow",
        "reverse",
        "override_aes",
        "order",
        "direction",
    )

    def __init__(
        self,
        title: Any = None,
        nrow: int | None = None,
        ncol: int | None = None,
        byrow: bool = False,
        reverse: bool = False,
        override_aes: Mapping[str, Any] | None = None,
        order: int = 0,
        direction: str | None = None,
    ) -> None:
        if direction not in _DIRECTIONS:
            raise ValueError(
                f"direction must be 'horizontal', 'vertical' or None, got {direction!r}"
            )
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            raise ValueError(f"order must be an integer >= 0, got {order!r}")
        self.title = title
        self.nrow = _pos_int(nrow, "nrow")
        self.ncol = _pos_int(ncol, "ncol")
        self.byrow = bool(byrow)
        self.reverse = bool(reverse)
        self.override_aes: dict[str, Any] = {
            standardise_aes_name(k): v for k, v in dict(override_aes or {}).items()
        }
        self.order = int(order)
        self.direction = direction

    def grid(self, n: int, direction: str) -> tuple[int, int]:
        """``(nrow, ncol)`` for ``n`` keys laid out in ``direction``."""
        n = max(n, 1)
        nrow, ncol = self.nrow, self.ncol
        if nrow is None and ncol is None:
            return (1, n) if direction == "horizontal" else (n, 1)
        if nrow is None:
            assert ncol is not None
            return -(-n // ncol), ncol
        if ncol is None:
            return nrow, -(-n // nrow)
        if nrow * ncol < n:
            raise ValueError(f"guide_legend: nrow * ncol ({nrow} * {ncol}) < number of keys ({n})")
        return nrow, ncol

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GuideLegend) and all(
            getattr(self, s) == getattr(other, s) for s in self.__slots__
        )

    def __hash__(self) -> int:
        return hash(repr(self))

    def __repr__(self) -> str:
        args = ", ".join(f"{s}={getattr(self, s)!r}" for s in self.__slots__)
        return f"guide_legend({args})"


def guide_legend(
    title: Any = None,
    nrow: int | None = None,
    ncol: int | None = None,
    byrow: bool = False,
    reverse: bool = False,
    override_aes: Mapping[str, Any] | None = None,
    order: int = 0,
    direction: str | None = None,
) -> GuideLegend:
    """Legend guide options.

    ``title`` overrides the scale/labs title. ``nrow``/``ncol`` lay the keys out
    in a grid (default: one column, or one row for horizontal legends); keys fill
    the grid column by column unless ``byrow=True``. ``reverse`` reverses the key
    order. ``override_aes`` sets aesthetics of the key glyphs only, e.g.
    ``{"alpha": 1, "size": 0.12}`` (a list value gives one value per key).
    ``order`` (``>= 1``) sorts legends: lower first, then the ``order=0`` ones in
    scale order. ``direction`` (``"horizontal"``/``"vertical"``) overrides
    ``theme(legend_direction=...)`` for this legend.

    Legends of different aesthetics merge into one (with composite keys) when
    their titles, breaks and labels match, as in ggplot2; the first legend's
    options win and ``override_aes`` are combined.
    """
    return GuideLegend(title, nrow, ncol, byrow, reverse, override_aes, order, direction)


def is_legend_spec(spec: Any) -> bool:
    """True if a guide spec (string or object) asks for a key legend."""
    return isinstance(spec, GuideLegend) or spec == "legend"


def legend_frame(content: Mobject, theme: Theme) -> Mobject | None:
    """The background / border rectangle round ``content`` (padded by
    ``theme.legend_margin``), or ``None`` when the theme asks for neither."""
    background = getattr(theme, "legend_background", None)
    border = getattr(theme, "legend_border", False)
    if background is None and not border:
        return None
    rect = SurroundingRectangle(content, buff=getattr(theme, "legend_margin", 0.15))
    if background is not None:
        rect.set_fill(background, opacity=getattr(theme, "legend_background_opacity", 1.0))
    else:
        rect.set_fill(opacity=0)
    if border:
        rect.set_stroke(
            color=theme.color_of("legend_border_color"),
            width=getattr(theme, "legend_border_width", 1.0),
            opacity=1.0,
        )
    else:
        rect.set_stroke(width=0, opacity=0)
    return rect


_TITLE_SIDES = {"top": UP, "bottom": DOWN, "left": LEFT, "right": RIGHT}


def place_legend_title(title: Mobject, body: Mobject, theme: Theme, buff: float) -> Mobject:
    """Put ``title`` ``buff`` away from ``body`` on the side given by
    ``theme.legend_title_position`` (default ``"top"``). Above or below, it is
    justified horizontally over the body by ``theme.legend_title_hjust`` (else
    ``legend_title_align``; 0 = left edges aligned); left or right, it is centred
    vertically on the body."""
    position = getattr(theme, "legend_title_position", "top") or "top"
    title.next_to(body, _TITLE_SIDES[position], buff=buff)
    if position in ("left", "right"):
        title.set_y(body.get_center()[1])
    else:
        hjust = theme.legend_title_hjust_value()
        left = body.get_left()[0] + hjust * (body.width - title.width)
        title.set_x(left, direction=LEFT)
    return title
