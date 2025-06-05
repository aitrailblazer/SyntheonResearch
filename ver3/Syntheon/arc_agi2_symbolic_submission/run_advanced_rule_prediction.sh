#!/bin/bash
# =============================================================================
# Advanced Preprocessing and Rule Prediction Workflow
# =============================================================================
# This script runs the complete workflow for improving rule prediction using
# advanced preprocessing data:
# 1. Generate enhanced XML with advanced preprocessing data
# 2. Update main.py to use the enhanced data
# 3. Run the main solver with improved rule prediction
#
# Author: Syntheon Development Team
# Date: May 30, 2025
# =============================================================================

set -e  # Exit on error

echo "======================================================================"
echo "         Advanced Preprocessing and Rule Prediction Workflow           "
echo "======================================================================"

# Check if files exist
if [ ! -f "precompute_advanced_preprocessing.py" ]; then
    echo "Error: precompute_advanced_preprocessing.py not found"
    exit 1
fi

if [ ! -f "integrate_advanced_preprocessing.py" ]; then
    echo "Error: integrate_advanced_preprocessing.py not found"
    exit 1
fi

if [ ! -f "main.py" ]; then
    echo "Error: main.py not found"
    exit 1
fi

# Step 1: Generate enhanced XML with advanced preprocessing data
echo "Step 1: Generating enhanced XML with advanced preprocessing data..."
python precompute_advanced_preprocessing.py --input input/arc_agi2_training_combined.xml --output input/arc_agi2_training_enhanced.xml

# Step 2: Update main.py to use the enhanced data
echo "Step 2: Updating main.py to use the enhanced data..."
python integrate_advanced_preprocessing.py

# Step 3: Run the main solver with improved rule prediction
echo "Step 3: Running the main solver with improved rule prediction..."
python main.py

echo "======================================================================"
echo "                        Workflow completed!                            "
echo "======================================================================"
echo "Results are available in syntheon_output.xml and syntheon_output.log"
echo "======================================================================"
