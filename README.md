
# **SyntheonResearch**

Syntheon is a purely symbolic solver for ARC-AGI tasks. It reads XML datasets and
applies deterministic rules to predict the outputs for test grids.

## **Syntheon ARC-AGI Symbolic Predictor**

**Syntheon** is a *purely symbolic* prediction system designed to solve ARC-AGI tasks
using deterministic reasoning, structured pattern recognition, and task-specific rule
learning. It operates without LLM dependencies, ensuring explainable, reproducible
predictions suitable for rigorous evaluation and competition-grade benchmarks.


---

### **Core Principles**

* **🧠 Pure Symbolic Reasoning**
  No learning from language models—Syntheon operates on interpretable symbolic logic.

* **⚙️ Advanced Preprocessing Integration**

  Fully utilizes structured XML metadata for each ARC task, enabling entropy profiling,
  grid fingerprinting, and directional analysis.

* **🔍 Task-Specific Symbolic Learning**
  Each `<arc_agi_task>` is treated independently. Rules are extracted solely from
  `training_examples`, never from `test_examples`.

* **🎨 KWIC Pattern Detection**
  Symbolic analysis of **K**ey **W**ords **I**n **C**ontext (KWIC): identifying
  color co-occurrence, adjacency, and propagation logic.



* **🔍 Task-Specific Symbolic Learning**
  Each `<arc_agi_task>` is treated independently.
   Rules are extracted solely from `training_examples`, never from `test_examples`.

* **🎨 KWIC Pattern Detection**
  Symbolic analysis of **K**ey **W**ords **I**n **C**ontext (KWIC): identifying color co-occurrence, adjacency,
   and propagation logic.


* **🔁 Symmetry-Guided Processing**

  Transformation strategies guided by mirror, rotational, and translational symmetry
  detection within input-output pairs.


---

### **Dataset Specification**

* Input file: `arc_agi2_training_enhanced.xml`
* Contains: **1000 `<arc_agi_task>` blocks**

  * Each task includes:

    * `<training_examples>` — used for symbolic rule extraction
    * `<test_examples>` — used *only* for output validation

---

### **Final Objective**

* Develop generalized symbolic rules from the training examples of each task.
* Use these rules to predict the correct `output` for each test input.
* *Do not use* `test_examples` outputs for rule learning—only for correctness verification after inference.

---


## Running the Solver

1. Ensure Python 3.x is installed.

2. From the repository root, set `PYTHONPATH=src` so Python can locate the `syntheon` package.
3. Run the solver on the training dataset:

```bash
PYTHONPATH=src python -m syntheon.main \
  arc_agi2_symbolic_submission/input/arc_agi2_training_enhanced.xml \
  predictions.json [--verbose] [--summary]


PYTHONPATH=src python -m syntheon.main \
  arc_agi2_symbolic_submission/input/arc_agi2_training_enhanced.xml \
  predictions.json --summary

PYTHONPATH=src python -m syntheon.main \
  arc_agi2_symbolic_submission/input/arc_agi2_training_enhanced.xml \
  predictions.json --verbose

This command parses the XML tasks, applies the current rule set, and writes predictions to `predictions.json`.
The `predictor` module now learns simple color mappings from training examples and applies them to the tests.
It scans each training pair cell by cell to build a dictionary of input–output color
transformations (e.g., `{1: 3, 2: 4}`). Conflicts stop the mapping from being used.
When predicting, this `ColorMapRule` is applied to every test grid so the same
transformations occur automatically.
The solver prints detailed logs describing the learned rules and shows the input grid, intermediate steps, predicted
output, and real output for each test example. At the end of the run it reports how many tasks were solved and the
overall accuracy.
Progress messages show which task is currently being solved, e.g., "Solving task 3/100 (task_id)".
Use `--verbose` to display detailed rule application logs.
The optional `--summary` flag prints a final PASS/FAIL report for every task.
Logs also indicate which rules solved each task and suggest new color mappings when predictions fail.

If you see no output, rerun with `--verbose` to ensure logging is displayed.

The solver prints detailed logs describing the learned rules and shows the input grid, intermediate steps, predicted
output, and real output for each test example. At the end of the run it reports how many tasks were solved and the
overall accuracy.

Progress messages show which task is currently being solved, e.g., "Solving task 3/100 (task_id)".
Use `--verbose` to display detailed rule application logs.
The optional `--summary` flag prints a final PASS/FAIL report for every task.


## Testing

To verify basic functionality:

1. Run the solver as shown above on a few tasks.
2. Inspect `predictions.json` to ensure it contains task IDs and predicted output grids.
3. Execute the unit tests to verify the predictor implementation:

```bash
PYTHONPATH=src pytest -q
```
