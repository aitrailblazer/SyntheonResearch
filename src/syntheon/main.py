"""Command-line entry point for running the Syntheon solver."""

import argparse
import logging
import json
from pathlib import Path

from ingest import load_tasks, Task
from predictor import SymbolicPredictor





def main() -> None:
    parser = argparse.ArgumentParser(description="Run Syntheon symbolic solver")
    parser.add_argument("xml", type=Path, help="Path to arc_agi2_training_enhanced.xml")
    parser.add_argument("output", type=Path, help="Output JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    tasks = load_tasks(str(args.xml))
    results = {}
    correct = 0
    total = 0
    for task in tasks:
        predictor = SymbolicPredictor()
        predictor.learn(task)
        expected = [ex.output_grid for ex in task.tests]
        preds = predictor.predict(task)
        results[task.id] = preds
        match = expected == preds
        total += 1
        if match:
            correct += 1
        logging.info("Task %s %s", task.id, "PASS" if match else "FAIL")
        if not match:
            logging.debug("expected=%s predicted=%s", expected, preds)

    accuracy = correct / total * 100 if total else 0
    logging.info("Solved %d/%d tasks (%.2f%% accuracy)", correct, total, accuracy)
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
