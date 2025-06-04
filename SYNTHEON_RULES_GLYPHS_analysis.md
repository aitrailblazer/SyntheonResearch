# syntheon_rules_glyphs.xml Analysis

The `syntheon_rules_glyphs.xml` file defines a symbolic scroll named `SyntheonSymbolicRules` (version 1.3).
It enumerates a canonical glyph index followed by a library of transformation rules for ARC grids.

## Glyph Index
The file introduces numeric glyph codes mapped to unicode characters, providing compact notation for rule chains.
Each glyph also carries a weight used for meta-logic or selection.

## Core Rules
A selection of notable rules includes:
- **TilePatternExpansion** (R21) &mdash; expands a 2&times;2 tile into a 6&times;6 grid by repeated tiling.
- **MirrorBandExpansion** (R03) &mdash; mirrors each row to double the width of the grid.
- **ColorReplacement** (R31) &mdash; replaces every occurrence of a color with another color.
- **DiagonalFlip** (R33) &mdash; transposes the grid along its main diagonal.
- **CompleteSymmetry** (R27) &mdash; detects the strongest symmetry axis and completes the pattern accordingly.
- **SequentialRuleApplication** (R41) &mdash; allows chaining of two rules in sequence.
- **ConditionalRuleSwitch** (R42) &mdash; applies one of two rules based on a predicate.

The XML emphasizes composability by including `glyph_chain` strings and pseudo code for deterministic implementation.
