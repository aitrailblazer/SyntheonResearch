#!/bin/bash
# ============================================================================
# Comprehensive Performance Testing for Enhanced Rule Prediction
# ============================================================================
# This script evaluates the performance of the enhanced rule prediction system
# by running a series of tests against the ARC-AGI2 evaluation dataset.
#
# Author: Syntheon Development Team
# Date: May 30, 2025
# ============================================================================

echo "🚀 Starting comprehensive performance testing for enhanced rule prediction..."
echo "======================================================================"

# Configure test environment
export PYTHONPATH=$(pwd):$PYTHONPATH
TEST_OUTPUT_DIR="./test_results_$(date +"%Y%m%d_%H%M%S")"
mkdir -p $TEST_OUTPUT_DIR

# Run unit tests first
echo "📋 Running unit tests for enhanced parameter extraction..."
python3 tests_enhanced_extraction.py

# Check if unit tests passed
if [ $? -ne 0 ]; then
    echo "❌ Unit tests failed. Please fix the issues before continuing."
    exit 1
fi
echo "✅ Unit tests passed successfully."
echo

# Function to run test with different configurations
run_test() {
    CONFIG_NAME=$1
    PREPROCESSING_LEVEL=$2
    RULE_PRIORITY_METHOD=$3
    OUTPUT_FILE="$TEST_OUTPUT_DIR/${CONFIG_NAME}_results.json"
    
    echo "🔬 Running test with configuration: $CONFIG_NAME"
    echo "   - Preprocessing level: $PREPROCESSING_LEVEL"
    echo "   - Rule priority method: $RULE_PRIORITY_METHOD"
    
    # Run the test
    python3 main.py \
        --input="./input/scroll-arcagi2/arc_agi2_evaluation_combined.xml" \
        --preprocessing-level=$PREPROCESSING_LEVEL \
        --rule-priority=$RULE_PRIORITY_METHOD \
        --output=$OUTPUT_FILE
    
    # Check results
    SUCCESS_COUNT=$(grep -c "\"success\": true" $OUTPUT_FILE)
    TOTAL_COUNT=$(grep -c "\"task_id\":" $OUTPUT_FILE)
    SUCCESS_RATE=$(echo "scale=2; $SUCCESS_COUNT / $TOTAL_COUNT * 100" | bc)
    
    echo "   📊 Results: $SUCCESS_COUNT/$TOTAL_COUNT tasks solved (${SUCCESS_RATE}%)"
    echo
    
    # Save configuration details
    echo "{\"config_name\": \"$CONFIG_NAME\", \"preprocessing_level\": \"$PREPROCESSING_LEVEL\", \"rule_priority\": \"$RULE_PRIORITY_METHOD\", \"success_count\": $SUCCESS_COUNT, \"total_count\": $TOTAL_COUNT, \"success_rate\": $SUCCESS_RATE}" > "$TEST_OUTPUT_DIR/${CONFIG_NAME}_summary.json"
}

# Run tests with different configurations
echo "📋 Running comprehensive performance tests..."
echo "======================================================================"

# Baseline test (without enhanced features)
run_test "baseline" "basic" "default" 

# Test with only enhanced parameter extraction
run_test "enhanced_parameters" "advanced" "default"

# Test with only adaptive rule prioritization
run_test "adaptive_priority" "basic" "adaptive"

# Test with full enhanced system
run_test "full_enhanced" "advanced" "adaptive"

# Generate comparison report
echo "📊 Generating comparison report..."
python3 - <<EOF
import json
import os

# Load results
results = []
for filename in os.listdir("$TEST_OUTPUT_DIR"):
    if filename.endswith("_summary.json"):
        with open(os.path.join("$TEST_OUTPUT_DIR", filename), 'r') as f:
            results.append(json.load(f))

# Sort by success rate
results.sort(key=lambda x: x['success_rate'], reverse=True)

# Print comparison table
print("\n====================================================================")
print("📊 Performance Comparison")
print("====================================================================")
print(f"{'Configuration':<20} {'Preprocessing':<15} {'Rule Priority':<15} {'Success Rate':<15}")
print("-" * 70)
for r in results:
    print(f"{r['config_name']:<20} {r['preprocessing_level']:<15} {r['rule_priority']:<15} {r['success_rate']}%")
print("====================================================================")

# Save the best configuration
with open("$TEST_OUTPUT_DIR/best_config.json", 'w') as f:
    json.dump(results[0], f, indent=2)
    
print(f"\n✅ Best configuration: {results[0]['config_name']} with {results[0]['success_rate']}% success rate")
print(f"   - Preprocessing level: {results[0]['preprocessing_level']}")
print(f"   - Rule priority method: {results[0]['rule_priority']}")
EOF

echo "======================================================================"
echo "✅ Performance testing completed. Results are available in: $TEST_OUTPUT_DIR"
echo "======================================================================"
