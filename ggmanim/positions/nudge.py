"""``position_nudge``: shift every row by a constant offset (ggplot2 semantics).

Mostly useful for labels next to points. The offsets are in *transformed* units
(on ``scale_x_log10`` a nudge of ``0.1`` is a factor of ~1.26) and are added to
every x-like (``x``, ``xmin``, ``xmax``, ``xend`` ...) / y-like column.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import Position
from .jitter import shift_positions

__all__ = ["PositionNudge", "position_nudge"]


class PositionNudge(Position):
    """Shift all positions by ``(x, y)``."""

    name = "nudge"

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        super().__init__(x=x, y=y)

    def compute_layer(
        self, data: pd.DataFrame, params: dict[str, Any], scales: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        if len(data) == 0 or (params.get("x", 0) == 0 and params.get("y", 0) == 0):
            return data
        return shift_positions(data, params.get("x", 0.0), params.get("y", 0.0))

    def compute_panel(
        self, data: pd.DataFrame, params: dict[str, Any], scales: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        return self.compute_layer(data, params, scales)


def position_nudge(x: float = 0.0, y: float = 0.0) -> PositionNudge:
    """Nudge positions by a fixed offset (ggplot2's ``position_nudge``), in
    transformed data units."""
    return PositionNudge(x=x, y=y)
