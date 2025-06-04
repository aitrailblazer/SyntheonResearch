from typing import List


class Rule:
    """Base class for symbolic rules."""

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        """Apply the rule to the grid and return the transformed grid."""
        raise NotImplementedError


class ColorReplacementRule(Rule):
    def __init__(self, src: int, dst: int):
        self.src = src
        self.dst = dst

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        return [[self.dst if c == self.src else c for c in row] for row in grid]


PRIMARY_RULES = {}
SECONDARY_RULES = {}


def register_primary(name: str, rule: Rule) -> None:
    PRIMARY_RULES[name] = rule


def register_secondary(name: str, rule: Rule) -> None:
    SECONDARY_RULES[name] = rule
