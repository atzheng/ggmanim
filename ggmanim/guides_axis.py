"""Axis guide options (ggplot2's ``guide_axis``).

:func:`guide_axis` configures how a position axis draws its tick labels when
they are crowded: ``check_overlap=True`` drops labels that would overlap an
already drawn one (trying the first, last and middle labels first, then the
middles of the halves, as ggplot2 does); ``n_dodge=n`` staggers the labels into
``n`` rows. Pass it to a position scale::

    scale_x_discrete(guide=guide_axis(n_dodge=2))
    scale_x_continuous(guide=guide_axis(check_overlap=True))

(``guides(x=guide_axis(...))`` is not supported yet: the axis is drawn from the
scale alone.) This module is pure Python; the drawing lives in
:mod:`ggmanim.coords`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["GuideAxis", "guide_axis", "axis_label_priority", "is_axis_spec", "check_axis_overlap"]


class GuideAxis:
    """Axis options (see :func:`guide_axis`)."""

    __slots__ = ("check_overlap", "n_dodge")

    def __init__(self, check_overlap: bool = False, n_dodge: int = 1) -> None:
        if isinstance(n_dodge, bool) or not isinstance(n_dodge, int) or n_dodge < 1:
            raise ValueError(f"n_dodge must be a positive integer, got {n_dodge!r}")
        self.check_overlap = bool(check_overlap)
        self.n_dodge = int(n_dodge)

    def __repr__(self) -> str:
        return f"guide_axis(check_overlap={self.check_overlap!r}, n_dodge={self.n_dodge!r})"


def guide_axis(check_overlap: bool = False, n_dodge: int = 1, **kwargs: Any) -> GuideAxis:
    """Axis guide options.

    Parameters
    ----------
    check_overlap : drop tick labels that overlap a label already drawn. Labels are
        considered in ggplot2's priority order (first, last, middle, then the
        middles of the remaining halves, recursively), so the ends survive.
    n_dodge : number of rows (x axis) or columns (y axis) the labels are staggered
        into: label ``i`` goes to row ``i % n_dodge``, rows stack away from the axis.
        ``check_overlap`` is applied to each row separately.

    ggplot2's dotted spellings (``check.overlap``, ``n.dodge``) are accepted.
    """
    if "check.overlap" in kwargs:
        check_overlap = kwargs.pop("check.overlap")
    if "n.dodge" in kwargs:
        n_dodge = kwargs.pop("n.dodge")
    if kwargs:
        raise TypeError(f"guide_axis() got unexpected arguments {sorted(kwargs)}")
    return GuideAxis(check_overlap, n_dodge)


def is_axis_spec(spec: Any) -> bool:
    """Is ``spec`` a :func:`guide_axis` object?"""
    return isinstance(spec, GuideAxis)


def axis_label_priority(n: int) -> list[int]:
    """0-based drawing order of ``n`` labels under ``check_overlap`` (ggplot2's
    ``axis_label_priority``): first, last, then the middle and, recursively, the
    middles of both halves."""
    if n <= 0:
        return []
    if n == 1:
        return [0]

    def between(x: int, y: int) -> list[int]:
        m = y - x + 1
        if m <= 2:
            return []
        mid = x - 1 + (m + 1) // 2
        return [mid, *between(x, mid), *between(mid, y)]

    return [0, n - 1, *(i - 1 for i in between(1, n))]


def check_axis_overlap(boxes: Sequence[tuple[float, float, float, float] | None]) -> list[bool]:
    """Which labels survive ``check_overlap``. ``boxes`` holds one
    ``(x0, y0, x1, y1)`` per label (``None`` for an empty label, always kept);
    labels are visited in :func:`axis_label_priority` order and dropped when their box
    intersects (touching edges do not count) one already kept."""
    keep = [True] * len(boxes)
    kept: list[tuple[float, float, float, float]] = []
    for i in axis_label_priority(len(boxes)):
        box = boxes[i]
        if box is None:
            continue
        if any(box[0] < k[2] and k[0] < box[2] and box[1] < k[3] and k[1] < box[3] for k in kept):
            keep[i] = False
        else:
            kept.append(box)
    return keep
