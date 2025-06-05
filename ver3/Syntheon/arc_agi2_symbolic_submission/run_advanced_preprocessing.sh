#!/bin/bash
# =================================================================
# Advanced Preprocessing Launcher Script
# =================================================================
# This script runs the advanced preprocessing precomputation on
# the ARC-AGI2 XML files and updates the main.py to use the
# enhanced XML file.
# =================================================================

# Set default values
INPUT_XML="input/arc_agi2_training_combined.xml"
OUTPUT_XML="input/arc_agi2_training_enhanced.xml"
VERBOSE="--verbose"

# Print banner
echo "============================================================="
echo "  Advanced Preprocessing System (7-Component) Precomputation"
echo "============================================================="
echo "This script will enrich the ARC-AGI2 XML with results from:"
echo "1. Structural Signature Analysis (SSA)"
echo "2. Scalability Potential Analysis (SPA)"
echo "3. Pattern Composition Decomposition (PCD)"
echo "4. Transformation Type Prediction (TTP)" 
echo "5. Geometric Invariant Analysis (GIA)"
echo "6. Multi-Scale Pattern Detection (MSPD)"
echo "7. Contextual Rule Prioritization (CRP)"
echo "============================================================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.6 or higher."
    exit 1
fi

# Check if the precomputation script exists
if [ ! -f "precompute_advanced_preprocessing.py" ]; then
    echo "Error: precompute_advanced_preprocessing.py not found."
    exit 1
fi

# Check if the input XML exists
if [ ! -f "$INPUT_XML" ]; then
    echo "Error: Input XML file not found: $INPUT_XML"
    exit 1
fi

# Ask for confirmation
echo ""
echo "This will process all examples in $INPUT_XML"
echo "and save results to $OUTPUT_XML"
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Make the precomputation script executable
chmod +x precompute_advanced_preprocessing.py

# Run the precomputation
echo ""
echo "Starting precomputation..."
python precompute_advanced_preprocessing.py --input "$INPUT_XML" --output "$OUTPUT_XML" $VERBOSE

# Check if precomputation was successful
if [ $? -ne 0 ]; then
    echo "Error: Precomputation failed."
    exit 1
fi

# Update main.py to use the enhanced XML
echo ""
echo "Updating main.py to use the enhanced XML..."

# Use sed to update the DATA_XML line in main.py
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s|DATA_XML = \"input/arc_agi2_training_combined.xml\"|DATA_XML = \"$OUTPUT_XML\"|g" main.py
else
    # Linux
    sed -i "s|DATA_XML = \"input/arc_agi2_training_combined.xml\"|DATA_XML = \"$OUTPUT_XML\"|g" main.py
fi

echo ""
echo "============================================================="
echo "  Advanced Preprocessing Integration Complete!"
echo "============================================================="
echo "The enhanced XML file has been created at: $OUTPUT_XML"
echo "main.py has been updated to use the enhanced XML."
echo ""
echo "You can now run main.py to use the Advanced Preprocessing"
echo "System with precomputed results for optimal performance."
echo "============================================================="
