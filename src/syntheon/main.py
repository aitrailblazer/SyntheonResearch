import argparse
import json
from pathlib import Path
from typing import List

from .ingest import load_tasks, Task
from .rule_engine import Rule


def predict(task: Task, rules: List[Rule]) -> List[List[List[int]]]:
    predictions = []
    for grid in task.test_inputs:
        result = grid
        for rule in rules:
            result = rule.apply(result)
        predictions.append(result)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Syntheon symbolic solver")
    parser.add_argument("xml", type=Path, help="Path to arc_agi2_training_enhanced.xml")
    parser.add_argument("output", type=Path, help="Output JSON file")
    args = parser.parse_args()

    tasks = load_tasks(str(args.xml))
    results = {}
    for task in tasks:
        preds = predict(task, [])
        results[task.task_id] = preds
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
