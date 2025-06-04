
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
2. Install dependencies with `pip install numpy` if not already available.
3. From the repository root, set `PYTHONPATH=src` so Python can locate the `syntheon` package.
4. Run the solver on the training dataset:


This command parses the XML tasks, applies the current rule set, and writes predictions to `predictions.json`.

## Testing

To verify basic functionality:

1. Run the solver as shown above on a few tasks.
2. Inspect `predictions.json` to ensure it contains task IDs and predicted output grids.
