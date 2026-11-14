"""Position adjustments (identity, stack, dodge, jitter ...).

See :mod:`ggmanim.positions.base` for the contract with the build pipeline and
the registry used by ``geom_*(position="...")``. The export block below is
generated from each module's ``__all__`` by ``scripts/gen_exports.py``.
"""

# -- generated exports (scripts/gen_exports.py) --
from .base import (
    POSITIONS,
    Position,
    PositionIdentity,
    get_position,
    parse_position,
    position_identity,
)
from .dodge import PositionDodge, PositionDodge2, position_dodge, position_dodge2
from .jitter import PositionJitter, PositionJitterdodge, position_jitter, position_jitterdodge
from .nudge import PositionNudge, position_nudge
from .stack import PositionFill, PositionStack, position_fill, position_stack

__all__ = [
    "POSITIONS",
    "Position",
    "PositionDodge",
    "PositionDodge2",
    "PositionFill",
    "PositionIdentity",
    "PositionJitter",
    "PositionJitterdodge",
    "PositionNudge",
    "PositionStack",
    "get_position",
    "parse_position",
    "position_dodge",
    "position_dodge2",
    "position_fill",
    "position_identity",
    "position_jitter",
    "position_jitterdodge",
    "position_nudge",
    "position_stack",
]
# -- end generated --
