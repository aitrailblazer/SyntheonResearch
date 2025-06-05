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

from typing import Dict, List, Optional, Type

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
        """Parse an XML rule scroll and store metadata and register simple rules."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for rule_el in root.findall("rule"):
            name = rule_el.get("name", "")
            self.metadata[name] = {
                "id": rule_el.get("id", ""),
                "desc": rule_el.findtext("description", default=""),
                "cond": rule_el.findtext("condition", default=""),
            }

            impl_cls = DEFAULT_XML_RULES.get(name)
            if impl_cls and name not in self.registry:
                self.register(name, impl_cls())

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

class ReflectVerticalRule(Rule):
    """Mirror the grid vertically (flip up/down)."""

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        return list(reversed(grid))


class RotatePatternRule(Rule):
    """Rotate the grid by 90°, 180° or 270° clockwise."""

    def __init__(self, degrees: int = 90) -> None:
        assert degrees in (90, 180, 270)
        self.degrees = degrees

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        if self.degrees == 180:
            return [list(reversed(row)) for row in reversed(grid)]

        h, w = len(grid), len(grid[0])
        if self.degrees == 90:
            return [[grid[h - j - 1][i] for j in range(h)] for i in range(w)]

        # 270 degrees
        return [[grid[j][w - i - 1] for j in range(h)] for i in range(w)]


class Rotate90Rule(RotatePatternRule):
    """Backward compatible 90° rotation rule."""

    def __init__(self) -> None:
        super().__init__(90)



class NullRule(Rule):
    """Placeholder rule used when an XML rule has no implementation."""

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        return grid


class CropToBoundingBoxRule(Rule):
    """Crop grid to the bounding box of all nonzero cells."""

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        rows = [i for i, row in enumerate(grid) if any(c != 0 for c in row)]
        cols = [j for j in range(len(grid[0])) if any(row[j] != 0 for row in grid)]
        if not rows or not cols:
            return grid
        r0, r1 = rows[0], rows[-1] + 1
        c0, c1 = cols[0], cols[-1] + 1
        return [row[c0:c1] for row in grid[r0:r1]]


class RemoveObjectsRule(Rule):
    """Remove connected components of a given color."""

    def __init__(self, color: int) -> None:
        self.color = color

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        h, w = len(grid), len(grid[0])
        visited = [[False] * w for _ in range(h)]

        def neighbors(r: int, c: int) -> List[tuple[int, int]]:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    yield nr, nc

        for i in range(h):
            for j in range(w):
                if grid[i][j] != self.color or visited[i][j]:
                    continue
                comp = [(i, j)]
                visited[i][j] = True
                q = [(i, j)]
                while q:
                    r, c = q.pop()
                    for nr, nc in neighbors(r, c):
                        if grid[nr][nc] == self.color and not visited[nr][nc]:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                            comp.append((nr, nc))
                for r, c in comp:
                    grid[r][c] = 0
        return grid


class ReplaceBorderWithColorRule(Rule):
    """Set the outer border of the grid to the given color."""

    def __init__(self, color: int) -> None:
        self.color = color

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        h, w = len(grid), len(grid[0])
        for c in range(w):
            grid[0][c] = self.color
            grid[h - 1][c] = self.color
        for r in range(h):
            grid[r][0] = self.color
            grid[r][w - 1] = self.color
        return grid


class DuplicateRowsOrColumnsRule(Rule):
    """Duplicate each row or column ``n`` times along the given axis."""

    def __init__(self, axis: int, n: int) -> None:
        assert axis in (0, 1)
        self.axis = axis
        self.n = max(1, n)

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        if self.axis == 0:
            return [row for row in grid for _ in range(self.n)]
        new_grid = []
        for row in grid:
            new_row: List[int] = []
            for cell in row:
                new_row.extend([cell] * self.n)
            new_grid.append(new_row)
        return new_grid


class MirrorBandExpansionRule(Rule):
    """Mirror each row to double the width of the grid."""

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        return [row + list(reversed(row)) for row in grid]


class FrameFillConvergenceRule(Rule):
    """Fill the center with one color and border with another."""

    def __init__(self, center_color: int = 1, border_color: int = 3) -> None:
        self.center_color = center_color
        self.border_color = border_color

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        h, w = len(grid), len(grid[0])
        out = [[self.center_color for _ in range(w)] for _ in range(h)]
        for c in range(w):
            out[0][c] = self.border_color
            out[h - 1][c] = self.border_color
        for r in range(h):
            out[r][0] = self.border_color
            out[r][w - 1] = self.border_color
        return out


class MajorityFillRule(Rule):
    """Fill background cells with the most common non-background color."""

    def __init__(self, background_color: int = 0) -> None:
        self.background_color = background_color

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        counts: Dict[int, int] = {}
        for row in grid:
            for c in row:
                if c != self.background_color:
                    counts[c] = counts.get(c, 0) + 1
        if not counts:
            majority = self.background_color
        else:
            majority = max(counts, key=counts.get)
        return [
            [majority if c == self.background_color else c for c in row]
            for row in grid
        ]


class FillHolesRule(Rule):
    """Fill enclosed zero regions with the surrounding color."""

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        h, w = len(grid), len(grid[0])
        visited = [[False] * w for _ in range(h)]

        def neighbors(r: int, c: int):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    yield nr, nc

        for i in range(h):
            for j in range(w):
                if grid[i][j] != 0 or visited[i][j]:
                    continue
                comp = [(i, j)]
                visited[i][j] = True
                q = [(i, j)]
                touches_border = i == 0 or j == 0 or i == h - 1 or j == w - 1
                while q:
                    r, c = q.pop()
                    for nr, nc in neighbors(r, c):
                        if grid[nr][nc] == 0 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                            comp.append((nr, nc))
                        elif grid[nr][nc] != 0:
                            pass
                        if nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1:
                            touches_border = True
                if not touches_border:
                    color_counts: Dict[int, int] = {}
                    for r, c in comp:
                        for nr, nc in neighbors(r, c):
                            col = grid[nr][nc]
                            if col != 0:
                                color_counts[col] = color_counts.get(col, 0) + 1
                    if color_counts:
                        fill_color = max(color_counts, key=color_counts.get)
                        for r, c in comp:
                            grid[r][c] = fill_color
        return grid


class ObjectCountingRule(Rule):
    """Return a 1×1 grid containing the number of nonzero objects."""

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        h, w = len(grid), len(grid[0])
        visited = [[False] * w for _ in range(h)]

        def neighbors(r: int, c: int):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    yield nr, nc

        count = 0
        for i in range(h):
            for j in range(w):
                if grid[i][j] == 0 or visited[i][j]:
                    continue
                count += 1
                q = [(i, j)]
                visited[i][j] = True
                while q:
                    r, c = q.pop()
                    for nr, nc in neighbors(r, c):
                        if grid[nr][nc] != 0 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            q.append((nr, nc))
        return [[count]]


class ColorSwappingRule(Rule):
    """Swap two colors throughout the grid."""

    def __init__(self, color_a: int, color_b: int) -> None:
        self.color_a = color_a
        self.color_b = color_b

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        return [
            [
                self.color_b if c == self.color_a else self.color_a if c == self.color_b else c
                for c in row
            ]
            for row in grid
        ]


class ScalePattern2xRule(Rule):
    """Scale the grid by a factor of 2."""

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        return [
            [cell for cell in row for _ in range(2)]
            for row in grid
            for _ in range(2)
        ]


class ScalePattern3xRule(Rule):
    """Scale the grid by a factor of 3."""

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        return [
            [cell for cell in row for _ in range(3)]
            for row in grid
            for _ in range(3)
        ]


class ScalePatternHalfRule(Rule):
    """Scale the grid by 0.5 using simple subsampling."""

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        h, w = len(grid), len(grid[0])
        new_h = max(1, h // 2)
        new_w = max(1, w // 2)
        return [[grid[i * 2][j * 2] for j in range(new_w)] for i in range(new_h)]


class TilePatternExpansionRule(Rule):
    """Repeat a 2×2 tile to produce a 6×6 grid."""

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        if len(grid) != 2 or len(grid[0]) != 2:
            return grid
        out: List[List[int]] = []
        for _ in range(3):
            for row in grid:
                out.append(row * 3)
        return out



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


# Mapping of rule names from ``syntheon_rules_glyphs.xml`` to built-in
# implementations. Unlisted rules default to :class:`NullRule` when loaded
# via :meth:`RuleEngine.load_rules_metadata`.
DEFAULT_XML_RULES: Dict[str, Type[Rule]] = {
    "DiagonalFlip": DiagonalFlipRule,
    "ReflectHorizontal": HorizontalMirrorRule,
    "ReflectVertical": ReflectVerticalRule,
    "RotatePattern": RotatePatternRule,
    "CropToBoundingBox": CropToBoundingBoxRule,
    "ColorReplacement": ColorReplacementRule,
    "RemoveObjects": RemoveObjectsRule,
    "ReplaceBorderWithColor": ReplaceBorderWithColorRule,
    "DuplicateRowsOrColumns": DuplicateRowsOrColumnsRule,
    "MirrorBandExpansion": MirrorBandExpansionRule,
    "FrameFillConvergence": FrameFillConvergenceRule,
    "MajorityFill": MajorityFillRule,
    "FillHoles": FillHolesRule,
    "ObjectCounting": ObjectCountingRule,
    "ColorSwapping": ColorSwappingRule,
    "ScalePattern2x": ScalePattern2xRule,
    "ScalePattern3x": ScalePattern3xRule,
    "ScalePatternHalf": ScalePatternHalfRule,
    "TilePatternExpansion": TilePatternExpansionRule,


}


