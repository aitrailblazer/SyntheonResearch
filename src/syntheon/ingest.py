from dataclasses import dataclass
from typing import List, Tuple
import xml.etree.ElementTree as ET


@dataclass
class Example:
    input_grid: List[List[int]]
    output_grid: List[List[int]]


@dataclass
class Task:
    task_id: str
    training_examples: List[Example]
    test_inputs: List[List[List[int]]]


def _parse_grid(element: ET.Element) -> List[List[int]]:
    return [list(map(int, row.text.split())) for row in element.findall("row")]


def load_tasks(xml_path: str) -> List[Task]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tasks = []
    for task_elem in root.findall("arc_agi_task"):
        tid = task_elem.get("id")
        train_examples = []
        for ex in task_elem.find("training_examples").findall("example"):
            input_grid = _parse_grid(ex.find("input"))
            output_grid = _parse_grid(ex.find("output"))
            train_examples.append(Example(input_grid, output_grid))
        test_inputs = []
        for ex in task_elem.find("test_examples").findall("example"):
            test_inputs.append(_parse_grid(ex.find("input")))
        tasks.append(Task(tid, train_examples, test_inputs))
    return tasks
