# ARC-AGI 2025 Project Specification

This document outlines the key requirements and datasets for the ARC-AGI 2025 challenge.

## Dataset Overview
- **Training data**: `arc_agi2_symbolic_submission/arc-prize-2025/arc-agi_training_challenges.json`
  and `arc-agi_training_solutions.json`.
- **Test data**: `arc-agi_test_challenges.json` with separate solution files for evaluation.
- **Evaluation data**: `arc-agi_evaluation_challenges.json` and `arc-agi_evaluation_solutions.json`
  define the leaderboard benchmark.

## Submission Format
- Predictions should be output as JSON matching the structure of `sample_submission.json`.
- Each entry maps a task identifier to a symbolic grid or color output as required by the challenge.
- Solutions must be deterministic and reproducible with no dependence on machine learning models.

## Scoring Criteria
- Accuracy is measured by exact match between predicted grids and the
  corresponding solutions in the evaluation set.
- Tie-breaking favors smaller solution code and clearer documentation of symbolic reasoning steps.

## Timeline
- **Dataset release**: Early 2025.
- **Submission deadline**: Mid 2025 with multiple evaluation rounds.
- **Final results**: Announced toward the end of 2025 after manual review of top submissions.

This specification complements the roadmap and emphasizes the use of purely symbolic rule systems for all
predictions.
