
# ARC Symbolic Predictor Package (Final)

## Contents

* `syntheon_rules.xml` – symbolic scroll with glyph weights and three example rules.
* `syntheon_engine.py` – minimal engine to parse the scroll and execute demo rules.
* `main.py` – CLI utility that:
    * loads training XML `arc_agi2_training_combined.xml`
    * iterates through each training example
    * applies a selected rule (default: `TilePatternExpansion`)
    * writes:
      * `syntheon_output.log` – run‑time log
      * `syntheon_output.xml` – verbose results for **every** example
      * `syntheon_solutions.xml` – *only* the examples predicted correctly
    * prints a concise verdict to stdout

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

Place `arc_agi2_training_combined.xml` in the same directory before running.

## Extending
* Add new rules to `syntheon_rules.xml`.
* Implement corresponding logic in `syntheon_engine.py::_apply_rule`.
* Update `main.py` to choose rule dynamically (e.g., based on `task_id`).

## Note
This is a toy starter showcasing how XML‑driven rules *could* be linked
to concrete logic. Only simple tiling & mirroring rules are implemented.
