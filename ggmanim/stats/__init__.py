"""Statistical transformations (pure pandas/numpy; no manim imports)."""

from .base import GROUPING_AES, NO_GROUP, Stat, add_group, get_stat, is_discrete, scale_limits
from .function import StatFunction, stat_function
from .identity import StatIdentity, stat_identity
from .smooth import StatSmooth, stat_smooth

#: name -> Stat class, used by :func:`get_stat` to resolve ``stat="smooth"``.
STATS = {
    "identity": StatIdentity,
    "smooth": StatSmooth,
    "function": StatFunction,
}

__all__ = [
    "Stat",
    "StatIdentity",
    "StatSmooth",
    "StatFunction",
    "stat_identity",
    "stat_smooth",
    "stat_function",
    "add_group",
    "get_stat",
    "is_discrete",
    "scale_limits",
    "NO_GROUP",
    "GROUPING_AES",
    "STATS",
]
