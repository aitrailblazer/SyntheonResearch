#!/usr/bin/env python3
"""
Advanced Preprocessing Precomputation Script for ARC-AGI2
=========================================================

This script enriches the ARC-AGI2 XML files with precomputed results from the
Advanced Preprocessing System (7-component):

1. Structural Signature Analysis (SSA) - analyzes size, symmetry, color patterns
2. Scalability Potential Analysis (SPA) - evaluates scaling potential
3. Pattern Composition Decomposition (PCD) - detects repeating units
4. Transformation Type Prediction (TTP) - predicts transformation types
5. Geometric Invariant Analysis (GIA) - analyzes geometric constraints
6. Multi-Scale Pattern Detection (MSPD) - hierarchical pattern analysis
7. Contextual Rule Prioritization (CRP) - confidence-based rule ranking

This script reads the existing XML files, runs the Advanced Preprocessing System
on all input grids, and adds the results as XML nodes, similar to how KWIC
data is stored. The enriched XML can then be used by the main solver.

Usage:
    python precompute_advanced_preprocessing.py [--input INPUT_XML] [--output OUTPUT_XML]

Example:
    python precompute_advanced_preprocessing.py --input input/arc_agi2_training_combined.xml --output input/arc_agi2_training_enhanced.xml

Author: Syntheon Development Team
Date: May 30, 2025
"""

import os
import sys
import time
import argparse
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import asdict
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Import the Advanced Preprocessing System
try:
    from advanced_preprocessing_specification import AdvancedInputPreprocessor
    logging.info("✅ Advanced preprocessing modules loaded successfully")
except ImportError as e:
    logging.error(f"❌ Failed to import Advanced Preprocessing System: {e}")
    logging.error("Please ensure advanced_preprocessing_specification.py is in the current directory.")
    sys.exit(1)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Precompute Advanced Preprocessing results for ARC-AGI2 XML files")
    parser.add_argument("--input", default="input/arc_agi2_training_combined.xml", help="Input XML file path")
    parser.add_argument("--output", default="input/arc_agi2_training_enhanced.xml", help="Output XML file path")
    parser.add_argument("--tasks", help="Comma-separated list of task IDs to process (if omitted, all tasks are processed)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()

def extract_grid_from_element(element) -> List[List[int]]:
    """Extract grid data from an XML element"""
    grid = []
    for row_el in element.findall("row"):
        row_values = [int(v) for v in row_el.text.strip().split()]
        grid.append(row_values)
    return grid

def create_advanced_preprocessing_element(results, input_grid=None, prefix="advanced"):
    """
    Create an XML element containing the Advanced Preprocessing results
    
    Args:
        results: PreprocessingResults object from AdvancedInputPreprocessor
        input_grid: Original input grid (for fallback dimensions)
        prefix: XML element prefix
        
    Returns:
        ET.Element containing the results
    """
    # Create the root element for advanced preprocessing
    advanced_el = ET.Element(f"{prefix}_preprocessing")
    
    # Add overall confidence
    advanced_el.set("confidence", f"{results.overall_confidence:.4f}")
    advanced_el.set("completeness", f"{results.analysis_completeness:.4f}")
    
    # Add processing time if available
    if hasattr(results, "processing_time"):
        advanced_el.set("processing_time", f"{results.processing_time:.4f}")
    
    # Add structural signature
    if results.structural_signature:
        sig = results.structural_signature
        sig_el = ET.SubElement(advanced_el, f"{prefix}_signature")
        
        # Add basic grid properties
        if hasattr(sig, 'dimensions') and sig.dimensions:
            height, width = sig.dimensions
            sig_el.set("height", str(height))
            sig_el.set("width", str(width))
        else:
            # Extract dimensions from input grid if available
            grid_height = len(input_grid) if input_grid else 0
            grid_width = len(input_grid[0]) if input_grid and input_grid else 0
            sig_el.set("height", str(grid_height))
            sig_el.set("width", str(grid_width))
            
        # Add unique colors if available
        if hasattr(sig, 'unique_colors'):
            sig_el.set("unique_colors", str(sig.unique_colors))
        
        # Add size class if available
        if hasattr(sig, 'size_class'):
            sig_el.set("size_class", str(sig.size_class.name) if hasattr(sig.size_class, 'name') else str(sig.size_class))
            
        # Add total cells if available
        if hasattr(sig, 'total_cells'):
            sig_el.set("total_cells", str(sig.total_cells))
        
        # Add symmetry profile
        if hasattr(sig, "symmetry_profile") and sig.symmetry_profile:
            sym_el = ET.SubElement(sig_el, "symmetry")
            
            # Safely access symmetry scores
            if hasattr(sig.symmetry_profile, "horizontal_score"):
                sym_el.set("horizontal", f"{sig.symmetry_profile.horizontal_score:.4f}")
            else:
                sym_el.set("horizontal", "0.0000")
                
            if hasattr(sig.symmetry_profile, "vertical_score"):
                sym_el.set("vertical", f"{sig.symmetry_profile.vertical_score:.4f}")
            else:
                sym_el.set("vertical", "0.0000")
                
            if hasattr(sig.symmetry_profile, "diagonal_score"):
                sym_el.set("diagonal", f"{sig.symmetry_profile.diagonal_score:.4f}")
            else:
                sym_el.set("diagonal", "0.0000")
                
            # Try to get overall symmetry score
            if hasattr(sig.symmetry_profile, "symmetry_score") and callable(getattr(sig.symmetry_profile, "symmetry_score")):
                sym_el.set("overall", f"{sig.symmetry_profile.symmetry_score():.4f}")
            else:
                # Calculate average if method not available
                h_score = getattr(sig.symmetry_profile, "horizontal_score", 0.0)
                v_score = getattr(sig.symmetry_profile, "vertical_score", 0.0)
                d_score = getattr(sig.symmetry_profile, "diagonal_score", 0.0)
                avg_score = (h_score + v_score + d_score) / 3.0
                sym_el.set("overall", f"{avg_score:.4f}")
    
    # Add transformation predictions
    if results.transformation_predictions:
        pred_el = ET.SubElement(advanced_el, f"{prefix}_predictions")
        
        # Add top predictions
        for i, pred in enumerate(results.transformation_predictions):
            p_el = ET.SubElement(pred_el, "prediction")
            
            # Safely get transformation type
            if hasattr(pred, "transformation_type"):
                if hasattr(pred.transformation_type, "value"):
                    p_el.set("type", str(pred.transformation_type.value))
                else:
                    p_el.set("type", str(pred.transformation_type))
            else:
                p_el.set("type", "unknown")
                
            # Safely get confidence
            if hasattr(pred, "confidence"):
                p_el.set("confidence", f"{pred.confidence:.4f}")
            else:
                p_el.set("confidence", "0.5000")
                
            p_el.set("rank", str(i+1))
            
            # Add parameters if available
            if hasattr(pred, "parameters") and pred.parameters:
                param_el = ET.SubElement(p_el, "parameters")
                for key, value in pred.parameters.items():
                    # Convert complex values to string
                    if isinstance(value, (list, tuple, dict)):
                        value = json.dumps(value)
                    param_el.set(key, str(value))
    
    # Add rule prioritization
    if hasattr(results, "rule_prioritization") and results.rule_prioritization:
        rule_el = ET.SubElement(advanced_el, f"{prefix}_rules")
        
        # Add primary rules
        primary_el = ET.SubElement(rule_el, "primary_rules")
        for rule in results.rule_prioritization.get('primary_rules', []):
            r_el = ET.SubElement(primary_el, "rule")
            r_el.set("name", rule)
            if rule in results.rule_prioritization.get('rule_confidence', {}):
                r_el.set("confidence", f"{results.rule_prioritization['rule_confidence'][rule]:.4f}")
            else:
                r_el.set("confidence", "0.7000")  # Default confidence for primary rules
        
        # Add secondary rules
        secondary_el = ET.SubElement(rule_el, "secondary_rules")
        for rule in results.rule_prioritization.get('secondary_rules', []):
            r_el = ET.SubElement(secondary_el, "rule")
            r_el.set("name", rule)
            if rule in results.rule_prioritization.get('rule_confidence', {}):
                r_el.set("confidence", f"{results.rule_prioritization['rule_confidence'][rule]:.4f}")
            else:
                r_el.set("confidence", "0.5000")  # Default confidence for secondary rules
    
    # Add pattern composition results
    if hasattr(results, "pattern_composition") and results.pattern_composition:
        patterns_el = ET.SubElement(advanced_el, f"{prefix}_patterns")
        
        # Add detected patterns
        if 'repeating_units' in results.pattern_composition:
            for i, unit in enumerate(results.pattern_composition['repeating_units']):
                if not isinstance(unit, dict):
                    # Skip non-dict units
                    continue
                    
                unit_el = ET.SubElement(patterns_el, "unit")
                unit_el.set("size_h", str(unit.get('height', 0)))
                unit_el.set("size_w", str(unit.get('width', 0)))
                unit_el.set("confidence", f"{unit.get('confidence', 0):.4f}")
                unit_el.set("index", str(i+1))
                
        # Add summary stats if available
        if 'summary' in results.pattern_composition:
            summary = results.pattern_composition['summary']
            if isinstance(summary, dict):
                summary_el = ET.SubElement(patterns_el, "summary")
                for key, value in summary.items():
                    if isinstance(value, (int, float, str, bool)):
                        summary_el.set(key, str(value))
    
    return advanced_el

def process_task(task_element, preprocessor, args):
    """Process a single task element and add advanced preprocessing data"""
    task_id = task_element.get("id", "unknown")
    logging.info(f"Processing task {task_id}...")
    
    # Process each example in the task
    examples_processed = 0
    
    try:
        # Process training examples
        for example_kind in ["training_examples", "test_examples"]:
            examples_section = task_element.find(example_kind)
            if examples_section is None:
                continue
                
            for example in examples_section.findall("example"):
                example_index = example.get("index", "0")
                logging.info(f"  Processing {example_kind.split('_')[0]} example {example_index}...")
                
                # Process input
                input_element = example.find("input")
                if input_element is not None:
                    # Extract the input grid
                    input_grid = extract_grid_from_element(input_element)
                    
                    # Skip if grid is empty
                    if not input_grid or not input_grid[0]:
                        logging.warning(f"  Skipping empty grid in {task_id} example {example_index}")
                        continue
                    
                    try:
                        # Apply the Advanced Preprocessing System
                        start_time = time.time()
                        preprocessing_results = preprocessor.analyze_comprehensive_input(input_grid)
                        processing_time = time.time() - start_time
                        
                        # Add processing time to results
                        preprocessing_results.processing_time = processing_time
                        
                        # Create XML element with results
                        advanced_el = create_advanced_preprocessing_element(
                            preprocessing_results, 
                            input_grid=input_grid
                        )
                        
                        # Add the element to the input
                        input_element.append(advanced_el)
                        
                        # Log success
                        top_prediction = "none"
                        confidence = 0.0
                        if preprocessing_results.transformation_predictions:
                            if hasattr(preprocessing_results.transformation_predictions[0].transformation_type, 'value'):
                                top_prediction = preprocessing_results.transformation_predictions[0].transformation_type.value
                            else:
                                top_prediction = str(preprocessing_results.transformation_predictions[0].transformation_type)
                            confidence = preprocessing_results.transformation_predictions[0].confidence
                        
                        logging.info(f"  ✅ {task_id}#{example_index} - "
                                    f"Top prediction: {top_prediction} ({confidence:.2f}) - "
                                    f"Time: {processing_time:.2f}s")
                        
                        examples_processed += 1
                        
                    except Exception as e:
                        logging.error(f"  ❌ Failed to process {task_id}#{example_index}: {str(e)}")
                        # Add fallback minimal preprocessing element
                        try:
                            fallback_el = ET.SubElement(input_element, "advanced_preprocessing")
                            fallback_el.set("confidence", "0.0000")
                            fallback_el.set("completeness", "0.0000")
                            fallback_el.set("error", str(e)[:100])  # Truncate long error messages
                            
                            # Add empty signature with grid dimensions
                            sig_el = ET.SubElement(fallback_el, "advanced_signature")
                            sig_el.set("height", str(len(input_grid)))
                            sig_el.set("width", str(len(input_grid[0]) if input_grid else 0))
                            sig_el.set("unique_colors", str(len(set(sum(input_grid, [])))))
                            
                            logging.info(f"  ℹ️ Added fallback preprocessing element for {task_id}#{example_index}")
                        except Exception as fallback_e:
                            logging.error(f"  ⚠️ Even fallback preprocessing failed: {str(fallback_e)}")
    except Exception as e:
        logging.error(f"❌ Failed to process task {task_id}: {str(e)}")
    
    return examples_processed

def main():
    """Main execution function"""
    args = parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Parse task filter if provided
    task_filter = None
    if args.tasks:
        task_filter = set(args.tasks.split(","))
        logging.info(f"Processing only tasks: {', '.join(task_filter)}")
    
    # Initialize the Advanced Preprocessing System
    try:
        preprocessor = AdvancedInputPreprocessor(
            enable_caching=True,
            analysis_depth="deep"
        )
        logging.info("Initialized Advanced Preprocessing System")
    except Exception as e:
        logging.error(f"Failed to initialize preprocessor: {e}")
        sys.exit(1)
    
    # Parse the XML file
    try:
        logging.info(f"Parsing input file: {args.input}")
        tree = ET.parse(args.input)
        root = tree.getroot()
        logging.info(f"Successfully parsed XML with {len(root)} elements")
    except Exception as e:
        logging.error(f"Failed to parse XML: {e}")
        sys.exit(1)
    
    # Process all tasks
    total_tasks = 0
    total_examples = 0
    start_time = time.time()
    
    for task in root.findall("arc_agi_task"):
        task_id = task.get("id", "unknown")
        
        # Skip if task_filter is set and this task is not in it
        if task_filter and task_id not in task_filter:
            continue
            
        examples_processed = process_task(task, preprocessor, args)
        total_examples += examples_processed
        total_tasks += 1
    
    total_time = time.time() - start_time
    
    # Log summary
    logging.info(f"\n==== Preprocessing Summary ====")
    logging.info(f"Total tasks processed: {total_tasks}")
    logging.info(f"Total examples processed: {total_examples}")
    logging.info(f"Total processing time: {total_time:.2f}s")
    logging.info(f"Average time per example: {total_time/total_examples:.2f}s")
    
    # Save the enriched XML
    try:
        logging.info(f"Saving enriched XML to: {args.output}")
        tree.write(args.output, encoding="utf-8", xml_declaration=True)
        logging.info(f"✅ Successfully saved enriched XML")
    except Exception as e:
        logging.error(f"Failed to save XML: {e}")
        sys.exit(1)
    
    logging.info("Done!")

if __name__ == "__main__":
    main()
