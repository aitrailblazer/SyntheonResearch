"""Command-line entry point for running the Syntheon solver."""

import argparse
import logging
import json
from pathlib import Path

from ingest import load_tasks, Task
from predictor import SymbolicPredictor, suggest_color_map_rule





def main() -> None:
    parser = argparse.ArgumentParser(description="Run Syntheon symbolic solver")
    parser.add_argument("xml", type=Path, help="Path to arc_agi2_training_enhanced.xml")
    parser.add_argument("output", type=Path, help="Output JSON file")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging with full rule application details",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary of each task result after solving",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s:%(message)s")

    tasks = load_tasks(str(args.xml))
    results = {}
    task_results = []
    correct = 0
    total = 0
    total_tasks = len(tasks)
    for idx, task in enumerate(tasks, start=1):
        logging.info("Solving task %d/%d (%s)", idx, total_tasks, task.id)
        predictor = SymbolicPredictor()
        predictor.learn(task)
        expected = [ex.output_grid for ex in task.tests]
        preds = predictor.predict(task)
        results[task.id] = preds
        match = expected == preds
        total += 1
        if match:
            correct += 1
        logging.info(
            "Task %s %s using rules %s",
            task.id,
            "PASS" if match else "FAIL",
            predictor.rules,
        )
        if not match:
            logging.debug("expected=%s predicted=%s", expected, preds)
            suggestion = suggest_color_map_rule(preds, expected)
            if suggestion:
                logging.info("Suggested new rule %s", suggestion)
        task_results.append((task.id, match, predictor.rules))

    accuracy = correct / total * 100 if total else 0
    if args.summary:
        for tid, ok, rules in task_results:
            logging.info("%s %s rules=%s", tid, "PASS" if ok else "FAIL", rules)
    logging.info(
        "Solved %d/%d tasks (%.2f%% accuracy)",
        correct,
        total,
        accuracy,
    )
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
