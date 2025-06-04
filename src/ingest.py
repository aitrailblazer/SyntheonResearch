from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
import xml.etree.ElementTree as ET


@dataclass
class Example:
    """Single ARC-AGI example containing input and output grids."""

    index: int
    input_grid: List[List[int]]
    output_grid: List[List[int]]

    def __repr__(self) -> str:  # pragma: no cover - simple representation
        cols = len(self.input_grid[0]) if self.input_grid else 0
        return f"Example(index={self.index}, input={len(self.input_grid)}x{cols})"


@dataclass
class Task:
    """ARC-AGI task with metadata, training examples and test examples."""

    id: str
    metadata_xml: str
    training: List[Example]
    tests: List[Example]


def _parse_grid(elem: ET.Element) -> List[List[int]]:
    grid = []
    for row in elem.findall("row"):
        if row.text:
            grid.append([int(x) for x in row.text.strip().split()])
    return grid


def _parse_example(elem: ET.Element) -> Example:
    idx = int(elem.attrib.get("index", 0))
    input_elem = elem.find("input")
    output_elem = elem.find("output")
    input_grid = _parse_grid(input_elem) if input_elem is not None else []
    output_grid = _parse_grid(output_elem) if output_elem is not None else []
    return Example(index=idx, input_grid=input_grid, output_grid=output_grid)


def load_tasks(xml_path: str | Path) -> List[Task]:
    """Parse an ARC-AGI XML file and return a list of tasks."""
    path = Path(xml_path)
    tree = ET.parse(path)
    root = tree.getroot()
    tasks: List[Task] = []
    for task_elem in root.findall("arc_agi_task"):
        task_id = task_elem.attrib.get("id", "")
        meta_elem = task_elem.find("metadata")
        metadata_xml = ET.tostring(meta_elem, encoding="unicode") if meta_elem is not None else ""
        train_examples = []
        te_elem = task_elem.find("training_examples")
        if te_elem is not None:
            for ex in te_elem.findall("example"):
                train_examples.append(_parse_example(ex))
        test_examples = []
        test_elem = task_elem.find("test_examples")
        if test_elem is not None:
            for ex in test_elem.findall("example"):
                test_examples.append(_parse_example(ex))
        tasks.append(Task(id=task_id, metadata_xml=metadata_xml, training=train_examples, tests=test_examples))
    return tasks


def _cli(argv: Iterable[str] | None = None) -> int:
    """Basic command-line interface for manual verification."""
    import argparse

    parser = argparse.ArgumentParser(description="Load ARC-AGI tasks from an XML file")
    parser.add_argument("xml", type=str, help="Path to ARC-AGI XML file")
    args = parser.parse_args(list(argv) if argv is not None else None)

    tasks = load_tasks(args.xml)
    print(f"Loaded {len(tasks)} tasks")
    if tasks:
        first = tasks[0]
        print(f"First task: id={first.id} training={len(first.training)} test={len(first.tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
