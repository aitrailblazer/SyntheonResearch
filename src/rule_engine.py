"""Simple rule engine with a registry for multi-tier rules.

This module defines a minimal :class:`Rule` interface along with placeholder
implementations.  Rules can be registered as either *primary* or *secondary*
which mirrors the multi-tier architecture described in ``ROADMAP.md``.

Example
-------
>>> from rule_engine import ColorReplacementRule, register_primary
>>> rule = ColorReplacementRule(1, 2)
>>> register_primary("replace_1_with_2", rule)
>>> rule.apply([[1, 0], [0, 1]])
[[2, 0], [0, 2]]
"""

from __future__ import annotations

from typing import Dict, List


class Rule:
    """Interface for all symbolic rules."""

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        """Apply the rule to a grid and return the transformed grid."""
        raise NotImplementedError


class ColorReplacementRule(Rule):
    """Replace every instance of ``src`` with ``dst``."""

    def __init__(self, src: int, dst: int) -> None:
        self.src = src
        self.dst = dst

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        return [[self.dst if c == self.src else c for c in row] for row in grid]


# registries for the multi-tier rule system
PRIMARY_RULES: Dict[str, Rule] = {}
SECONDARY_RULES: Dict[str, Rule] = {}


def register_primary(name: str, rule: Rule) -> None:
    """Register a rule as primary."""
    PRIMARY_RULES[name] = rule


def register_secondary(name: str, rule: Rule) -> None:
    """Register a rule as secondary."""
    SECONDARY_RULES[name] = rule
