#!/bin/bash
# ============================================================================
# Run Advanced Rule Prediction with Enhanced Parameters
# ============================================================================
# This script runs the ARC-AGI2 solver with the enhanced rule prediction system
# including adaptive rule prioritization and enhanced parameter extraction.
#
# Author: Syntheon Development Team
# Date: May 30, 2025
# ============================================================================

echo "🚀 Running ARC-AGI2 solver with enhanced rule prediction..."
echo "======================================================================"

# Configure environment
export PYTHONPATH=$(pwd):$PYTHONPATH
OUTPUT_FILE="submission_enhanced_$(date +"%Y%m%d_%H%M%S").json"

# Run the solver with enhanced configuration
python3 main.py \
    --input="./input/scroll-arcagi2/arc_agi2_evaluation_combined.xml" \
    --preprocessing-level="advanced" \
    --rule-priority="adaptive" \
    --output="$OUTPUT_FILE" \
    --verbose

# Report results
SUCCESS_COUNT=$(grep -c "\"success\": true" "$OUTPUT_FILE")
TOTAL_COUNT=$(grep -c "\"task_id\":" "$OUTPUT_FILE")
SUCCESS_RATE=$(echo "scale=2; $SUCCESS_COUNT / $TOTAL_COUNT * 100" | bc)

echo "======================================================================"
echo "✅ Solver run completed with enhanced rule prediction"
echo "📊 Results: $SUCCESS_COUNT/$TOTAL_COUNT tasks solved (${SUCCESS_RATE}%)"
echo "📋 Output file: $OUTPUT_FILE"
echo "======================================================================"
