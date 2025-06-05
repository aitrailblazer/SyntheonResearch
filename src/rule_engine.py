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

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

from ingest import Example, Task

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


    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ColorReplacementRule({self.src}->{self.dst})"


class ColorMapRule(Rule):
    """Map multiple colors according to a dictionary."""

    def __init__(self, mapping: Dict[int, int]) -> None:
        self.mapping = mapping

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        return [[self.mapping.get(c, c) for c in row] for row in grid]

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ColorMapRule({self.mapping})"


# registries for the multi-tier rule system
PRIMARY_RULES: Dict[str, Rule] = {}
SECONDARY_RULES: Dict[str, Rule] = {}


def register_primary(name: str, rule: Rule) -> None:
    """Register a rule as primary."""
    PRIMARY_RULES[name] = rule


def register_secondary(name: str, rule: Rule) -> None:
    """Register a rule as secondary."""
    SECONDARY_RULES[name] = rule

class RuleEngine:
    """Simple engine capable of loading and applying symbolic rules."""

    def __init__(self) -> None:
        self.registry: Dict[str, Rule] = {}
        self.metadata: Dict[str, Dict[str, str]] = {}

    # ------------------------------------------------------------------
    # Registration and lookup
    # ------------------------------------------------------------------
    def register(self, name: str, rule: Rule) -> None:
        """Register a rule implementation."""
        self.registry[name] = rule

    def available_rules(self) -> List[str]:  # pragma: no cover - simple helper
        return sorted(self.registry)

    # ------------------------------------------------------------------
    # XML loading
    # ------------------------------------------------------------------
    def load_rules_metadata(self, xml_path: str) -> None:
        """Parse an XML rule scroll and store metadata."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for rule_el in root.findall("rule"):
            name = rule_el.get("name", "")
            self.metadata[name] = {
                "id": rule_el.get("id", ""),
                "desc": rule_el.findtext("description", default=""),
                "cond": rule_el.findtext("condition", default=""),
            }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def apply(self, name: str, grid: List[List[int]], **kwargs) -> List[List[int]]:
        if name not in self.registry:
            raise KeyError(f"Unknown rule: {name}")
        rule = self.registry[name]
        # Rules may choose to ignore kwargs
        return rule.apply(grid, **kwargs) if hasattr(rule, "apply") else rule(grid, **kwargs)

    # ------------------------------------------------------------------
    # Learning and solving
    # ------------------------------------------------------------------
    def learn_rule(
        self, examples: List[Example], max_chain_length: int = 1
    ) -> Optional[Rule]:
        """Find a rule or rule chain that explains all training examples."""

        # Search single rules first
        for rule in self.registry.values():
            if all(rule.apply(ex.input_grid) == ex.output_grid for ex in examples):
                return rule

        # Optionally search chains of length two
        if max_chain_length >= 2:
            for r1 in self.registry.values():
                for r2 in self.registry.values():
                    chain = RuleChain(r1, r2)
                    if all(chain.apply(ex.input_grid) == ex.output_grid for ex in examples):
                        return chain

        return None

    def solve_task(self, task: Task, max_chain_length: int = 1) -> List[List[List[int]]]:
        """Learn from the task's training examples and predict test outputs."""

        rule = self.learn_rule(task.training, max_chain_length)
        results = []
        for ex in task.tests:
            grid = ex.input_grid
            if rule is not None:
                grid = rule.apply(grid)
            results.append(grid)
        return results


# --------------------------- Example primitives ---------------------------
class DiagonalFlipRule(Rule):
    """Transpose the grid along its main diagonal."""

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        return [list(row) for row in zip(*grid)]


class HorizontalMirrorRule(Rule):
    """Mirror the grid horizontally."""

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        return [list(reversed(row)) for row in grid]


class Rotate90Rule(Rule):
    """Rotate the grid 90 degrees clockwise."""

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        size = len(grid)
        return [
            [grid[size - j - 1][i] for j in range(size)]
            for i in range(size)
        ]


class RuleChain(Rule):
    """Apply a sequence of rules in order."""

    def __init__(self, *rules: Rule) -> None:
        self.rules = list(rules)

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        result = grid
        for rule in self.rules:
            result = rule.apply(result)
        return result

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"RuleChain({', '.join(repr(r) for r in self.rules)})"

