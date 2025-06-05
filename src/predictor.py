"""Simple symbolic predictor for ARC-AGI tasks."""

from __future__ import annotations

from typing import Dict, List, Optional
import logging

from ingest import Example, Task
from rule_engine import ColorMapRule, Rule


def learn_color_map_rule(examples: List[Example]) -> Optional[ColorMapRule]:
    """Derive a global color mapping from training examples."""
    mapping: Dict[int, int] = {}
    for ex in examples:
        logging.debug("Training example %s input=%s output=%s", ex.index, ex.input_grid, ex.output_grid)
        for in_row, out_row in zip(ex.input_grid, ex.output_grid):
            for src, dst in zip(in_row, out_row):
                if src == dst:
                    continue
                if src in mapping:
                    if mapping[src] != dst:
                        logging.debug("Conflicting mapping for color %s: %s vs %s", src, mapping[src], dst)
                        return None
                else:
                    mapping[src] = dst
    logging.info("Derived color map: %s", mapping)
    return ColorMapRule(mapping) if mapping else None



def suggest_color_map_rule(
    predicted: List[List[List[int]]], expected: List[List[List[int]]]
) -> Optional[ColorMapRule]:
    """Propose a color mapping to transform ``predicted`` into ``expected``."""
    mapping: Dict[int, int] = {}
    for pred_grid, exp_grid in zip(predicted, expected):
        for p_row, e_row in zip(pred_grid, exp_grid):
            for p, e in zip(p_row, e_row):
                if p == e:
                    continue
                if p in mapping:
                    if mapping[p] != e:
                        return None
                else:
                    mapping[p] = e
    return ColorMapRule(mapping) if mapping else None


class SymbolicPredictor:
    """Predictor that learns simple color mappings from training data."""

    def __init__(self) -> None:
        self.rules: List[Rule] = []

    def learn(self, task: Task) -> None:
        logging.info("Learning rules for task %s", task.id)
        rule = learn_color_map_rule(task.training)
        if rule:
            self.rules = [rule]
        else:
            self.rules = []
        logging.info("Learned rules: %s", self.rules)

    def predict(self, task: Task) -> List[List[List[int]]]:
        predictions = []
        for example in task.tests:
            logging.info("Test example %s input: %s", example.index, example.input_grid)
            grid = example.input_grid
            for rule in self.rules:
                logging.info("Applying rule %s", rule)
                grid = rule.apply(grid)
                logging.debug("Intermediate grid: %s", grid)
            predictions.append(grid)
            logging.info("Predicted: %s", grid)
            logging.info("Expected:  %s", example.output_grid)
        return predictions
