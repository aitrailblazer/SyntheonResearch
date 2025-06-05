"""Command-line entry point for running the Syntheon solver."""

import argparse
import logging
import json
from pathlib import Path
from typing import List

from ingest import load_tasks, Task
from rule_engine import Rule


def predict(task: Task, rules: List[Rule]) -> List[List[List[int]]]:
    """Apply all rules to each test grid and return the predicted grids."""
    predictions = []
    for example in task.tests:
        result = example.input_grid
        for rule in rules:
            result = rule.apply(result)
        predictions.append(result)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Syntheon symbolic solver")
    parser.add_argument("xml", type=Path, help="Path to arc_agi2_training_enhanced.xml")
    parser.add_argument("output", type=Path, help="Output JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    tasks = load_tasks(str(args.xml))
    results = {}
    correct = 0
    total = 0
    for task in tasks:
        expected = [ex.output_grid for ex in task.tests]
        preds = predict(task, [])
        results[task.id] = preds
        match = expected == preds
        total += 1
        if match:
            correct += 1
        logging.info("Task %s %s", task.id, "PASS" if match else "FAIL")
        if not match:
            logging.debug("expected=%s predicted=%s", expected, preds)

    accuracy = correct / total * 100 if total else 0
    logging.info("Accuracy %.2f%%", accuracy)
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
