# ARC AGI Task Schema

This document outlines the structure of each `<arc_agi_task>` block in `arc_agi2_training_enhanced.xml`.
It summarizes the main tags and attributes used to describe ARC tasks, metadata, and examples.

## Top-Level Structure

```xml
<arc_agi_tasks>
  <arc_agi_task id="...">
    <metadata>
      ...
    </metadata>
    <training_examples count="N">
      ...
    </training_examples>
    <test_examples count="M">
      ...
    </test_examples>
  </arc_agi_task>
  ...
</arc_agi_tasks>
```

## Metadata Section

- `<statistics>`: counts of colors, grid sizes, transformations, and color roles.
- `<kwic>` blocks: color co-occurrence statistics for each example. Multiple windows may appear.
For example, `training` and `training_outputs_consolidated` windows measure color adjacency in different contexts.

## Examples Section

Each `<training_examples>` or `<test_examples>` block contains `<example index="i">` elements.

### Example Details

```xml
<example index="0">
  <input height="H" width="W">
    <row index="0">...</row>
    ...
    <kwic ... />
    <advanced_preprocessing confidence="..." completeness="..." processing_time="...">
      <advanced_signature height="H" width="W" size_class="..." total_cells="...">
        <symmetry horizontal="..." vertical="..." diagonal="..." overall="..." />
      </advanced_signature>
      <advanced_predictions>
        <prediction type="..." confidence="..." rank="...">
          <parameters ... />
        </prediction>
        ...
      </advanced_predictions>
      <advanced_rules>
        <primary_rules>
          <rule name="..." confidence="..." />
          ...
        </primary_rules>
        <secondary_rules />
      </advanced_rules>
      <advanced_patterns />
    </advanced_preprocessing>
  </input>
  <output height="H2" width="W2">
    <row index="0">...</row>
    ...
    <kwic ... />
  </output>
</example>
```

The same structure is used for test examples, but their `output` sections are only for validation and must not
influence rule learning.

---

This schema can be used to parse the training file and build deterministic rules for the ARC AGI solver.
