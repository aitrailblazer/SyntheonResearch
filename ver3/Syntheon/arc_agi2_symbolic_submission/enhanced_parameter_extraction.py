#!/usr/bin/env python3
"""
Enhanced Parameter Extraction Utility for Advanced Preprocessing
===============================================================

This module provides advanced parameter extraction capabilities for the 
ARC-AGI2 symbolic solver, leveraging the 7-component preprocessing system.

Key features:
1. Specialized transformation type to rule mappings
2. Parameter extraction based on grid analysis
3. Rule chain recommendation with confidence scores
4. Adaptive parameter adjustment based on output validation

Usage:
    Import this module in main.py to use the parameter extraction functions.

Author: Syntheon Development Team
Date: June 12, 2025
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

def extract_transformation_parameters(advanced_preprocessing, rule_name, input_grid=None, output_grid=None):
    """
    Extract parameter suggestions for a specific rule based on transformation predictions.
    
    Args:
        advanced_preprocessing: Dictionary with advanced preprocessing data
        rule_name: Name of the rule to get parameters for
        input_grid: Optional input grid for additional parameter inference
        output_grid: Optional output grid for validation-based parameter enhancement
        
    Returns:
        Dictionary with parameter suggestions or empty dict if none available
    """
    if not advanced_preprocessing or 'predictions' not in advanced_preprocessing:
        return {}
    
    # ENHANCED: Expanded rule to transformation mappings with more specific patterns
    rule_to_transformation = {
        # Scaling rules
        "SimpleScaling": ["simple_scaling", "uniform_scaling", "magnification"],
        "GridRescaling": ["simple_scaling", "non_uniform_scaling", "asymmetric_scaling"],
        "UniformScaling": ["simple_scaling", "uniform_scaling", "symmetric_scaling"],
        
        # Tiling rules
        "TilePatternExpansion": ["tiling_with_mirroring", "pattern_completion", "pattern_repetition", "symmetric_expansion"],
        "MirrorBandExpansion": ["tiling_with_mirroring", "mirror_reflection", "band_expansion", "symmetric_replication"],
        "DuplicateRowsOrColumns": ["tiling_with_mirroring", "pattern_repetition", "row_column_duplication"],
        
        # Color transformation rules
        "ColorReplacement": ["color_transformation", "color_substitution", "color_remapping"],
        "ColorSwapping": ["color_transformation", "color_exchange", "color_permutation"],
        "MajorityFill": ["color_transformation", "pattern_completion", "color_majority_fill"],
        
        # Geometric transformation rules
        "RotateClockwise": ["geometric_transformation", "rotation", "clockwise_rotation"],
        "RotateCounterClockwise": ["geometric_transformation", "rotation", "counterclockwise_rotation"],
        "DiagonalFlip": ["geometric_transformation", "reflection", "diagonal_reflection"],
        "HorizontalFlip": ["geometric_transformation", "reflection", "horizontal_reflection"],
        "VerticalFlip": ["geometric_transformation", "reflection", "vertical_reflection"],
        "RotatePattern": ["geometric_transformation", "rotation", "pattern_rotation"],
        "ReflectHorizontal": ["geometric_transformation", "reflection", "horizontal_reflection", "mirror_horizontal"],
        "ReflectVertical": ["geometric_transformation", "reflection", "vertical_reflection", "mirror_vertical"],
        
        # ScalePattern variants
        "ScalePattern2x": ["simple_scaling", "uniform_scaling", "magnification", "double_size"],
        "ScalePattern3x": ["simple_scaling", "uniform_scaling", "magnification", "triple_size"],
        "ScalePatternHalf": ["simple_scaling", "reduction", "resolution_decrease", "half_size"],
        "CompleteSymmetry": ["geometric_transformation", "symmetry_detection", "pattern_completion", "auto_symmetry"],
        "ExtendPattern": ["pattern_completion", "pattern_extension", "sequence_continuation", "predictive_patterning"],
        "FillCheckerboard": ["pattern_completion", "checkerboard_pattern", "alternating_pattern", "binary_pattern"],
        "PatternRotation": ["geometric_transformation", "pattern_rotation", "orientation_change", "angular_transformation"],
        "PatternMirroring": ["geometric_transformation", "pattern_mirroring", "reflection", "symmetry_creation"],
        
        # Pattern completion rules
        "FillHoles": ["pattern_completion", "hole_filling", "gap_completion"],
        "FrameFillConvergence": ["pattern_completion", "frame_fill", "boundary_convergence"],
        "BorderCompletion": ["pattern_completion", "border_fill", "edge_completion"],
        "ReplaceBorderWithColor": ["pattern_completion", "border_modification", "edge_coloring"],
        
        # Object-based rules
        "ObjectCounting": ["complex_combination", "object_analysis", "enumeration"],
        "RemoveObjects": ["complex_combination", "object_removal", "filtering"],
        "CropToBoundingBox": ["complex_combination", "bounding_box", "cropping"],
        
        # Extended rules for enhanced mapping
        "FillByConnectivity": ["pattern_completion", "connectivity_filling", "region_growing"],
        "FloodFill": ["pattern_completion", "flood_filling", "seed_expansion"],
        "ExtendLines": ["pattern_completion", "line_extension", "continuity_preservation"],
        "HalvePattern": ["size_reduction", "pattern_halving", "shrinking"],
        "ApplyMask": ["complex_combination", "masking", "selective_application"],
        "GravityAdjust": ["complex_combination", "gravity_simulation", "falling_objects"],
        "Upscale": ["simple_scaling", "enlargement", "resolution_increase"],
        "Downscale": ["simple_scaling", "reduction", "resolution_decrease"]
    }
    
    # Get compatible transformation types
    compatible_types = rule_to_transformation.get(rule_name, [])
    if not compatible_types:
        return {}
    
    # ENHANCED: Use weighted scoring for prediction matching
    best_prediction = None
    best_score = 0.0
    
    for pred in advanced_preprocessing['predictions']:
        pred_type = pred.get('type', '')
        confidence = pred.get('confidence', 0.0)
        
        # Calculate match score based on prediction type and confidence
        if pred_type in compatible_types:
            # Primary match gets full confidence
            match_score = confidence
        elif any(pred_type in ct for ct in compatible_types):
            # Partial match gets reduced confidence
            match_score = confidence * 0.7
        else:
            # No match
            match_score = 0.0
            
        # Consider prediction rank (earlier predictions are better)
        rank_multiplier = 1.0
        if 'rank' in pred:
            rank = pred.get('rank', 99)
            rank_multiplier = max(0.5, 1.0 - (rank * 0.1))  # Decrease with rank, but not below 0.5
            
        final_score = match_score * rank_multiplier
        
        if final_score > best_score:
            best_prediction = pred
            best_score = final_score
    
    if not best_prediction or 'parameters' not in best_prediction or best_score < 0.1:
        # Special handling for ColorReplacement - try direct parameter extraction
        if rule_name == "ColorReplacement" and input_grid is not None and output_grid is not None:
            # Get unique colors in input grid
            input_colors = np.unique(input_grid)
            
            # Comprehensive color mapping detection
            for from_c in input_colors:
                for to_c in range(10):  # ARC uses colors 0-9
                    # Test if replacing from_c with to_c produces output
                    test_grid = input_grid.copy()
                    test_grid[test_grid == from_c] = to_c
                    if np.array_equal(test_grid, output_grid):
                        return {'from_color': int(from_c), 'to_color': int(to_c)}
            
            # Fallback: try simple direct mapping if colors changed
            changed_pixels = input_grid != output_grid
            if np.any(changed_pixels):
                changed_input_vals = input_grid[changed_pixels]
                changed_output_vals = output_grid[changed_pixels]
                if len(set(changed_input_vals)) == 1 and len(set(changed_output_vals)) == 1:
                    return {'from_color': int(changed_input_vals[0]), 'to_color': int(changed_output_vals[0])}
        
        return {}
    
    # Extract and convert parameters to appropriate types
    params = {}
    for key, value in best_prediction['parameters'].items():
        try:
            # Try as int
            params[key] = int(value)
        except ValueError:
            try:
                # Try as float
                params[key] = float(value)
            except ValueError:
                # Try as list/array (for parameters like unit_size)
                if value.startswith('[') and value.endswith(']'):
                    try:
                        # Parse array representation
                        array_str = value[1:-1].strip()
                        if ',' in array_str:
                            # Comma-separated format
                            params[key] = [int(x.strip()) for x in array_str.split(',')]
                        else:
                            # Space-separated format
                            params[key] = [int(x.strip()) for x in array_str.split()]
                    except ValueError:
                        # Keep as string if parsing fails
                        params[key] = value
                else:
                    # Keep as string for other cases
                    params[key] = value
    
    # ENHANCED: Add rule-specific parameter inference based on input/output grids
    if input_grid is not None:
        # Add general grid-based parameters
        params = enhance_parameters_with_grid_analysis(rule_name, params, input_grid, output_grid)
        
        # Add rule-specific parameter adjustments
        params = add_rule_specific_parameters(rule_name, params, input_grid, output_grid)
    
    # ENHANCED: Format parameters for compatibility with engine.apply_rule
    # Ensure parameters are in the correct format for the engine
    formatted_params = format_parameters_for_engine(rule_name, params)
    
    return formatted_params

def enhance_parameters_with_grid_analysis(rule_name, params, input_grid, output_grid=None):
    """
    Enhance parameters with grid-based analysis.
    
    Args:
        rule_name: Name of the rule
        params: Basic parameters from preprocessing
        input_grid: Input grid array
        output_grid: Optional output grid array for validation
        
    Returns:
        Enhanced parameters
    """
    # Get dimensions for size-based parameters
    input_shape = input_grid.shape
    output_shape = output_grid.shape if output_grid is not None else None
    
    # Handle scaling rules
    if rule_name in ["SimpleScaling", "GridRescaling", "UniformScaling", "Upscale", "Downscale", 
                     "ScalePattern2x", "ScalePattern3x", "ScalePatternHalf"]:
        if 'scale_factor' not in params and output_shape is not None:
            # Try to infer scale factor from grid sizes
            height_ratio = output_shape[0] / input_shape[0]
            width_ratio = output_shape[1] / input_shape[1]
            
            # For exact matches
            if abs(height_ratio - width_ratio) < 0.1 and abs(round(height_ratio) - height_ratio) < 0.1:
                params['scale_factor'] = int(round(height_ratio))
            
            # For non-uniform scaling
            elif rule_name == "GridRescaling":
                params['height_scale'] = int(round(height_ratio))
                params['width_scale'] = int(round(width_ratio))
        
        # Set default scale factors for ScalePattern rules
        elif rule_name == "ScalePattern2x" and 'scale_factor' not in params:
            params['scale_factor'] = 2
        elif rule_name == "ScalePattern3x" and 'scale_factor' not in params:
            params['scale_factor'] = 3
        elif rule_name == "ScalePatternHalf" and 'scale_factor' not in params:
            params['scale_factor'] = 0.5
    
    # Handle tiling rules
    elif rule_name in ["TilePatternExpansion", "MirrorBandExpansion", "DuplicateRowsOrColumns"]:
        # Try to infer unit size for tiling
        if 'unit_size' not in params:
            # Analyze grid to detect repeating patterns
            unit_h, unit_w = detect_repeating_unit(input_grid)
            if unit_h > 0 and unit_w > 0:
                params['unit_size'] = [unit_h, unit_w]
            elif min(input_shape) <= 5:  # Small grids likely use whole grid as unit
                params['unit_size'] = [int(input_shape[0]), int(input_shape[1])]
        
        # For MirrorBandExpansion, detect axis
        if rule_name == "MirrorBandExpansion" and 'axis' not in params:
            # Check if input is wider than tall (horizontal expansion) or vice versa
            if input_shape[1] > input_shape[0]:
                params['axis'] = 0  # Horizontal bands
            else:
                params['axis'] = 1  # Vertical bands
                
        # For DuplicateRowsOrColumns, detect axis and count
        if rule_name == "DuplicateRowsOrColumns":
            if 'axis' not in params and output_shape is not None:
                # Compare input and output shapes to determine duplication axis
                if output_shape[0] > input_shape[0] and output_shape[1] == input_shape[1]:
                    params['axis'] = 0  # Duplicate rows
                    params['n'] = output_shape[0] - input_shape[0]
                elif output_shape[1] > input_shape[1] and output_shape[0] == input_shape[0]:
                    params['axis'] = 1  # Duplicate columns
                    params['n'] = output_shape[1] - input_shape[1]
    
    # Handle color transformation rules
    elif rule_name in ["ColorReplacement", "ColorSwapping", "MajorityFill"]:
        # Get unique colors in input grid
        input_colors = np.unique(input_grid)
        
        if rule_name == "ColorReplacement" and output_grid is not None:
            # Try to detect color mapping by comparing input and output
            output_colors = np.unique(output_grid)
            
            # Comprehensive color mapping detection
            for from_c in input_colors:
                for to_c in range(10):  # ARC uses colors 0-9
                    # Test if replacing from_c with to_c produces output
                    test_grid = input_grid.copy()
                    test_grid[test_grid == from_c] = to_c
                    if np.array_equal(test_grid, output_grid):
                        params['from_color'] = int(from_c)
                        params['to_color'] = int(to_c)
                        return params  # Found valid mapping
            
            # Fallback: try simple direct mapping if colors changed
            changed_pixels = input_grid != output_grid
            if np.any(changed_pixels):
                changed_input_vals = input_grid[changed_pixels]
                changed_output_vals = output_grid[changed_pixels]
                if len(set(changed_input_vals)) == 1 and len(set(changed_output_vals)) == 1:
                    params['from_color'] = int(changed_input_vals[0])
                    params['to_color'] = int(changed_output_vals[0])
        
        elif rule_name == "MajorityFill":
            # Count occurrences of each color
            color_counts = {}
            for color in input_colors:
                color_counts[color] = np.sum(input_grid == color)
            
            # Find most common color
            majority_color = max(color_counts, key=color_counts.get)
            params['color'] = int(majority_color)
    
    # Handle geometric transformation rules
    elif rule_name in ["RotateClockwise", "RotateCounterClockwise", "RotatePattern"]:
        if rule_name == "RotateClockwise" and 'degrees' not in params:
            # Default to 90 degrees for clockwise rotation
            params['degrees'] = 90
        elif rule_name == "RotateCounterClockwise" and 'degrees' not in params:
            # Default to 90 degrees for counter-clockwise rotation
            params['degrees'] = 90
        elif rule_name == "RotatePattern" and output_grid is not None:
            # Try to detect rotation angle by comparing dimensions
            if input_shape == output_shape:
                # Same dimensions - could be 180 degrees
                params['degrees'] = 180
            elif input_shape[0] == output_shape[1] and input_shape[1] == output_shape[0]:
                # Swapped dimensions - could be 90 or 270 degrees
                params['degrees'] = 90  # Default to 90
    
    return params

def add_rule_specific_parameters(rule_name, params, input_grid, output_grid=None):
    """
    Add rule-specific parameter enhancements.
    
    Args:
        rule_name: Name of the rule
        params: Parameters from basic extraction
        input_grid: Input grid array
        output_grid: Optional output grid for validation
        
    Returns:
        Enhanced parameters with rule-specific additions
    """
    # Get unique colors from input
    input_colors = list(np.unique(input_grid))
    
    # Rule-specific parameter enhancements
    if rule_name == "FillHoles":
        # Determine background color if not provided
        if 'background_color' not in params:
            # Assume most common color is background
            color_counts = {}
            for color in input_colors:
                color_counts[color] = np.sum(input_grid == color)
            
            background_color = max(color_counts, key=color_counts.get)
            params['background_color'] = int(background_color)
    
    elif rule_name == "CropToBoundingBox":
        # No additional parameters needed, the rule detects non-zero elements automatically
        pass
    
    elif rule_name == "ReplaceBorderWithColor":
        # Determine replacement color if not provided
        if 'color' not in params and len(input_colors) > 1:
            # Use a color that's different from the border
            border_elements = []
            h, w = input_grid.shape
            
            # Get border elements
            border_elements.extend(input_grid[0, :].flatten())  # Top row
            border_elements.extend(input_grid[-1, :].flatten())  # Bottom row
            border_elements.extend(input_grid[1:-1, 0].flatten())  # Left column (excluding corners)
            border_elements.extend(input_grid[1:-1, -1].flatten())  # Right column (excluding corners)
            
            border_colors = set(border_elements)
            non_border_colors = [c for c in input_colors if c not in border_colors]
            
            if non_border_colors:
                params['color'] = int(non_border_colors[0])
            else:
                # If all colors appear in border, use first color
                params['color'] = int(input_colors[0])
    
    elif rule_name == "RemoveObjects":
        # For RemoveObjects, detect which color is likely the "object"
        if 'color' not in params and len(input_colors) > 1:
            # Assume least common color represents objects
            color_counts = {}
            for color in input_colors:
                color_counts[color] = np.sum(input_grid == color)
            
            object_color = min(color_counts, key=color_counts.get)
            params['color'] = int(object_color)
    
    elif rule_name == "ObjectCounting":
        # For ObjectCounting, identify the color to count
        if 'color' not in params and len(input_colors) > 1:
            # Try to identify a non-zero, non-background color
            if 0 in input_colors and len(input_colors) > 2:
                # Assume 0 is background, count the second most common color
                color_counts = {}
                for color in input_colors:
                    if color != 0:
                        color_counts[color] = np.sum(input_grid == color)
                
                if color_counts:
                    count_color = max(color_counts, key=color_counts.get)
                    params['color'] = int(count_color)
            else:
                # Default to the least common color
                color_counts = {}
                for color in input_colors:
                    color_counts[color] = np.sum(input_grid == color)
                
                count_color = min(color_counts, key=color_counts.get)
                params['color'] = int(count_color)
    
    return params

def format_parameters_for_engine(rule_name, params):
    """
    Format parameters for compatibility with engine.apply_rule.
    
    Args:
        rule_name: Name of the rule
        params: Extracted parameters
        
    Returns:
        Formatted parameters compatible with the engine
    """
    # Rules that need special parameter formatting
    if rule_name in ["ColorReplacement", "ColorSwapping"]:
        # Handle from_color/to_color vs color_a/color_b naming differences
        if 'from_color' in params and 'to_color' in params:
            if rule_name == "ColorReplacement":
                # Keep from_color/to_color for ColorReplacement
                pass
            elif rule_name == "ColorSwapping":
                # Convert to color_a/color_b for ColorSwapping
                params['color_a'] = params.pop('from_color')
                params['color_b'] = params.pop('to_color')
    
    # Handle unit_size parameter that needs to be converted to individual values
    if 'unit_size' in params and isinstance(params['unit_size'], list) and len(params['unit_size']) == 2:
        unit_size = params.pop('unit_size')
        params['unit_height'] = unit_size[0]
        params['unit_width'] = unit_size[1]
    
    return params

def detect_repeating_unit(grid):
    """
    Detect the size of a repeating unit in a grid.
    
    Args:
        grid: Input grid array
        
    Returns:
        Tuple (height, width) of the detected repeating unit, or (0, 0) if none found
    """
    if grid is None or grid.size == 0:
        return 0, 0
    
    h, w = grid.shape
    
    # Check for horizontal repetition
    for unit_w in range(1, w // 2 + 1):
        if w % unit_w == 0:
            is_repeating = True
            for i in range(unit_w, w, unit_w):
                if not np.array_equal(grid[:, 0:unit_w], grid[:, i:i+unit_w]):
                    is_repeating = False
                    break
            if is_repeating:
                return h, unit_w
    
    # Check for vertical repetition
    for unit_h in range(1, h // 2 + 1):
        if h % unit_h == 0:
            is_repeating = True
            for i in range(unit_h, h, unit_h):
                if not np.array_equal(grid[0:unit_h, :], grid[i:i+unit_h, :]):
                    is_repeating = False
                    break
            if is_repeating:
                return unit_h, w
    
    # Check for 2D repetition (both horizontal and vertical)
    for unit_h in range(1, h // 2 + 1):
        for unit_w in range(1, w // 2 + 1):
            if h % unit_h == 0 and w % unit_w == 0:
                is_repeating = True
                unit = grid[0:unit_h, 0:unit_w]
                
                for i in range(0, h, unit_h):
                    for j in range(0, w, unit_w):
                        if not np.array_equal(unit, grid[i:i+unit_h, j:j+unit_w]):
                            is_repeating = False
                            break
                    if not is_repeating:
                        break
                
                if is_repeating:
                    return unit_h, unit_w
    
    # No repetition found
    return 0, 0

def is_horizontally_symmetric(grid):
    """
    Check if a grid has horizontal symmetry (mirror image across horizontal axis).
    
    Args:
        grid: Input grid array
        
    Returns:
        Boolean indicating if the grid is horizontally symmetric
        Float symmetry score between 0.0 (not symmetric) and 1.0 (perfectly symmetric)
    """
    if grid is None or grid.size == 0:
        return False, 0.0
    
    h, w = grid.shape
    half_h = h // 2
    
    # For perfectly symmetric grids
    if h % 2 == 0:  # Even height
        top_half = grid[:half_h, :]
        bottom_half = np.flipud(grid[half_h:, :])
        perfect_match = np.array_equal(top_half, bottom_half)
        
        if perfect_match:
            return True, 1.0
        
        # Calculate symmetry score
        match_count = np.sum(top_half == bottom_half)
        total_cells = top_half.size
        symmetry_score = match_count / total_cells
        
        return symmetry_score > 0.8, symmetry_score
    else:  # Odd height
        top_half = grid[:half_h, :]
        bottom_half = np.flipud(grid[half_h+1:, :])
        perfect_match = np.array_equal(top_half, bottom_half)
        
        if perfect_match:
            return True, 1.0
        
        # Calculate symmetry score
        match_count = np.sum(top_half == bottom_half)
        total_cells = top_half.size
        symmetry_score = match_count / total_cells
        
        return symmetry_score > 0.8, symmetry_score

def is_vertically_symmetric(grid):
    """
    Check if a grid has vertical symmetry (mirror image across vertical axis).
    
    Args:
        grid: Input grid array
        
    Returns:
        Boolean indicating if the grid is vertically symmetric
        Float symmetry score between 0.0 (not symmetric) and 1.0 (perfectly symmetric)
    """
    if grid is None or grid.size == 0:
        return False, 0.0
    
    h, w = grid.shape
    half_w = w // 2
    
    # For perfectly symmetric grids
    if w % 2 == 0:  # Even width
        left_half = grid[:, :half_w]
        right_half = np.fliplr(grid[:, half_w:])
        perfect_match = np.array_equal(left_half, right_half)
        
        if perfect_match:
            return True, 1.0
        
        # Calculate symmetry score
        match_count = np.sum(left_half == right_half)
        total_cells = left_half.size
        symmetry_score = match_count / total_cells
        
        return symmetry_score > 0.8, symmetry_score
    else:  # Odd width
        left_half = grid[:, :half_w]
        right_half = np.fliplr(grid[:, half_w+1:])
        perfect_match = np.array_equal(left_half, right_half)
        
        if perfect_match:
            return True, 1.0
        
        # Calculate symmetry score
        match_count = np.sum(left_half == right_half)
        total_cells = left_half.size
        symmetry_score = match_count / total_cells
        
        return symmetry_score > 0.8, symmetry_score

def get_adaptive_rule_priority(advanced_preprocessing, input_grid=None):
    """
    Dynamically prioritize rules based on preprocessing results and grid properties.
    
    Args:
        advanced_preprocessing: Dictionary with advanced preprocessing data
        input_grid: Optional input grid for additional property analysis
        
    Returns:
        Ordered list of rule names prioritized for the current task
    """
    # Default rule priority for fallback
    default_rules = [
        "ColorReplacement",
        "DiagonalFlip",
        "MirrorBandExpansion",
        "FillHoles",
        "CropToBoundingBox",
        "ColorSwapping",
        "ReplaceBorderWithColor",
        "ObjectCounting",
        "RemoveObjects",
        "TilePatternExpansion",
        "FrameFillConvergence",
        "RotateClockwise",
        "RotateCounterClockwise",
        "HorizontalFlip",
        "VerticalFlip",
        "MajorityFill",
        "FillByConnectivity",
        "FloodFill",
        "ExtendLines",
        "HalvePattern"
    ]
    
    if not advanced_preprocessing or 'predictions' not in advanced_preprocessing:
        return default_rules
    
    # Extract primary rules from advanced preprocessing
    primary_rules = advanced_preprocessing.get('primary_rules', [])
    
    # Extract grid properties from signature if available
    signature = advanced_preprocessing.get('signature', {})
    height = signature.get('height', 0)
    width = signature.get('width', 0)
    unique_colors = signature.get('unique_colors', 0)
    
    # Extract symmetry information
    symmetry = signature.get('symmetry', {})
    horizontal_sym = symmetry.get('horizontal', 0.0)
    vertical_sym = symmetry.get('vertical', 0.0)
    diagonal_sym = symmetry.get('diagonal', 0.0)
    
    # Additional grid analysis if input grid is provided
    if input_grid is not None:
        # Check if grid has actual symmetry (sometimes preprocessing might miss it)
        h_sym, h_score = is_horizontally_symmetric(input_grid)
        v_sym, v_score = is_vertically_symmetric(input_grid)
        
        # Update symmetry scores with actual analysis
        horizontal_sym = max(horizontal_sym, h_score)
        vertical_sym = max(vertical_sym, v_score)
        
        # Get grid dimensions and color count from input grid
        height, width = input_grid.shape
        unique_colors = len(np.unique(input_grid))
    
    # Create weighted rule sets based on grid properties
    geometry_rules = [
        "RotateClockwise",
        "RotateCounterClockwise",
        "DiagonalFlip",
        "HorizontalFlip",
        "VerticalFlip",
        "RotatePattern",
        "ReflectHorizontal",
        "ReflectVertical",
        "CropToBoundingBox"
    ]
    
    color_rules = [
        "ColorReplacement",
        "ColorSwapping",
        "MajorityFill",
        "FillByConnectivity",
        "FloodFill",
        "ReplaceBorderWithColor"
    ]
    
    pattern_rules = [
        "TilePatternExpansion",
        "MirrorBandExpansion",
        "DuplicateRowsOrColumns",
        "FillHoles",
        "FrameFillConvergence",
        "BorderCompletion",
        "ExtendLines"
    ]
    
    object_rules = [
        "ObjectCounting",
        "RemoveObjects",
        "ApplyMask",
        "GravityAdjust"
    ]
    
    scaling_rules = [
        "SimpleScaling",
        "GridRescaling",
        "UniformScaling",
        "Upscale",
        "Downscale",
        "HalvePattern",
        "ScalePattern2x",
        "ScalePattern3x",
        "ScalePatternHalf"
    ]
    
    # Create prioritized rule list based on grid properties and predictions
    prioritized_rules = []
    
    # First add explicitly predicted rules from preprocessing
    for rule in primary_rules:
        if rule not in prioritized_rules:
            prioritized_rules.append(rule)
    
    # Prioritize based on symmetry
    if horizontal_sym > 0.7:
        for rule in ["HorizontalFlip", "MirrorBandExpansion", "ReflectHorizontal", "CompleteSymmetry", "ExtendPattern"]:
            if rule not in prioritized_rules:
                prioritized_rules.append(rule)
    
    if vertical_sym > 0.7:
        for rule in ["VerticalFlip", "MirrorBandExpansion", "ReflectVertical", "CompleteSymmetry", "ExtendPattern"]:
            if rule not in prioritized_rules:
                prioritized_rules.append(rule)
    
    if diagonal_sym > 0.7:
        for rule in ["DiagonalFlip", "RotatePattern", "CompleteSymmetry"]:
            if rule not in prioritized_rules:
                prioritized_rules.append(rule)
    
    # Prioritize based on grid size
    if height <= 3 and width <= 3:
        # Small grids often use simple transformations
        for rule in color_rules + geometry_rules:
            if rule not in prioritized_rules:
                prioritized_rules.append(rule)
    elif height > 10 or width > 10:
        # Large grids often use scaling or object-based rules
        for rule in scaling_rules + object_rules:
            if rule not in prioritized_rules:
                prioritized_rules.append(rule)
    
    # Prioritize specific scaling rules if we have examples with both input and output
    if 'examples' in advanced_preprocessing and advanced_preprocessing['examples']:
        examples = advanced_preprocessing['examples']
        for example in examples:
            if 'input' in example and 'output' in example:
                input_example = np.array(example['input'])
                output_example = np.array(example['output'])
                
                if input_example.size > 0 and output_example.size > 0:
                    in_h, in_w = input_example.shape
                    out_h, out_w = output_example.shape
                    
                    height_ratio = out_h / in_h if in_h > 0 else 1.0
                    width_ratio = out_w / in_w if in_w > 0 else 1.0
                    
                    # Check for uniform scaling (both dimensions have same ratio)
                    if abs(height_ratio - width_ratio) < 0.1:
                        # For 2x scaling
                        if abs(height_ratio - 2.0) < 0.1:
                            for rule in ["ScalePattern2x", "SimpleScaling"]:
                                if rule not in prioritized_rules:
                                    prioritized_rules.append(rule)
                        # For 3x scaling
                        elif abs(height_ratio - 3.0) < 0.1:
                            for rule in ["ScalePattern3x", "SimpleScaling"]:
                                if rule not in prioritized_rules:
                                    prioritized_rules.append(rule)
                        # For 0.5x scaling (halving)
                        elif abs(height_ratio - 0.5) < 0.1:
                            for rule in ["ScalePatternHalf", "Downscale"]:
                                if rule not in prioritized_rules:
                                    prioritized_rules.append(rule)
                # Only need to check the first example
                break
    
    # Prioritize based on color count
    if unique_colors <= 2:
        # Binary grids often use pattern and geometric rules
        for rule in pattern_rules + geometry_rules:
            if rule not in prioritized_rules:
                prioritized_rules.append(rule)
    elif unique_colors >= 6:
        # Many colors often indicate color transformations
        for rule in color_rules:
            if rule not in prioritized_rules:
                prioritized_rules.append(rule)
    
    # Add remaining rules that haven't been added yet
    for rule in default_rules:
        if rule not in prioritized_rules:
            prioritized_rules.append(rule)
    
    return prioritized_rules

def get_recommended_rule_chains(advanced_preprocessing):
    """
    Get recommended rule chains based on advanced preprocessing.
    
    Args:
        advanced_preprocessing: Dictionary with advanced preprocessing data
        
    Returns:
        List of recommended rule chains
    """
    if not advanced_preprocessing or 'predictions' not in advanced_preprocessing:
        return []
    
    # Basic rule chains based on transformation types
    transformation_chains = {
        "tiling_with_mirroring": [
            "TilePatternExpansion -> ColorReplacement", 
            "MirrorBandExpansion -> ColorSwapping",
            "DuplicateRowsOrColumns -> CropToBoundingBox",
            "TilePatternExpansion -> FillHoles",
            "MirrorBandExpansion -> ReplaceBorderWithColor"
        ],
        "simple_scaling": [
            "SimpleScaling -> MajorityFill",
            "GridRescaling -> FillHoles",
            "UniformScaling -> CropToBoundingBox",
            "SimpleScaling -> ColorReplacement",
            "GridRescaling -> ReplaceBorderWithColor",
            "ScalePattern2x -> ColorReplacement -> FillHoles",
            "ScalePattern3x -> CropToBoundingBox -> ObjectCounting",
            "ScalePatternHalf -> MajorityFill -> BorderCompletion"
        ],
        "color_transformation": [
            "ColorReplacement -> CropToBoundingBox",
            "ColorSwapping -> RemoveObjects",
            "MajorityFill -> FillHoles",
            "ColorReplacement -> ReplaceBorderWithColor",
            "ColorSwapping -> ObjectCounting"
        ],
        "geometric_transformation": [
            "DiagonalFlip -> ColorReplacement",
            "RotateClockwise -> ColorSwapping",
            "HorizontalFlip -> FillHoles",
            "VerticalFlip -> CropToBoundingBox",
            "RotatePattern -> ReplaceBorderWithColor",
            "ReflectHorizontal -> RemoveObjects -> CropToBoundingBox",
            "ReflectVertical -> ColorReplacement -> ObjectCounting",
            "CompleteSymmetry -> FillHoles -> CropToBoundingBox",
            "CompleteSymmetry -> ColorReplacement -> ObjectCounting",
            "ExtendPattern -> CropToBoundingBox",
            "ExtendPattern -> ColorReplacement -> ObjectCounting",
            "FillCheckerboard -> CropToBoundingBox",
            "FillCheckerboard -> ColorReplacement -> ObjectCounting"
        ],
        "pattern_completion": [
            "FillHoles -> ObjectCounting",
            "FrameFillConvergence -> BorderCompletion",
            "BorderCompletion -> ColorReplacement",
            "FillHoles -> CropToBoundingBox",
            "FrameFillConvergence -> RemoveObjects",
            "CompleteSymmetry -> MajorityFill -> CropToBoundingBox",
            "ExtendPattern -> FillHoles -> CropToBoundingBox",
            "ExtendPattern -> MajorityFill",
            "FillCheckerboard -> MajorityFill",
            "FillCheckerboard -> FillHoles -> BorderCompletion",
            "PatternRotation -> CompleteSymmetry",
            "PatternRotation -> CropToBoundingBox",
            "PatternMirroring -> CompleteSymmetry",
            "PatternMirroring -> FillHoles",
            "PatternRotation -> PatternMirroring",
            "ExtendPattern -> PatternRotation",
            "PatternMirroring -> ExtendPattern"
        ],
        "complex_combination": [
            "RemoveObjects -> CropToBoundingBox",
            "ObjectCounting -> ColorReplacement",
            "CropToBoundingBox -> FrameFillConvergence",
            "RemoveObjects -> FillHoles",
            "ObjectCounting -> ReplaceBorderWithColor"
        ],
        "pattern_repetition": [
            "DuplicateRowsOrColumns -> ColorReplacement",
            "TilePatternExpansion -> ObjectCounting",
            "DuplicateRowsOrColumns -> FillHoles",
            "TilePatternExpansion -> CropToBoundingBox"
        ],
        "rotation": [
            "RotateClockwise -> FillHoles",
            "RotateCounterClockwise -> ColorReplacement",
            "RotatePattern -> MajorityFill",
            "RotateClockwise -> RemoveObjects"
        ],
        "reflection": [
            "DiagonalFlip -> FillHoles",
            "HorizontalFlip -> ColorReplacement",
            "VerticalFlip -> ObjectCounting",
            "DiagonalFlip -> CropToBoundingBox"
        ]
    }
    
    # ENHANCED: Get top three transformation predictions with confidence weighting
    weighted_transformations = []
    for pred in advanced_preprocessing['predictions']:
        pred_type = pred.get('type', '')
        confidence = pred.get('confidence', 0.0)
        rank = pred.get('rank', 99)
        
        # Create a score that considers both confidence and rank
        score = confidence * (1.0 - (rank * 0.05))
        weighted_transformations.append((pred_type, score))
    
    # Sort by score and take top 3
    weighted_transformations.sort(key=lambda x: x[1], reverse=True)
    top_transformations = [t[0] for t in weighted_transformations[:3]]
    
    # Build recommended chains
    recommended_chains = []
    for trans_type in top_transformations:
        # Add chains for exact matches
        if trans_type in transformation_chains:
            recommended_chains.extend(transformation_chains[trans_type])
        
        # Add chains for partial matches
        for key in transformation_chains:
            if trans_type in key or key in trans_type:
                recommended_chains.extend(transformation_chains[key])
    
    # ENHANCED: Add chains based on primary rules
    primary_rules = advanced_preprocessing.get('primary_rules', [])
    if primary_rules:
        # Generate chains using primary rules as first rule
        for rule in primary_rules[:2]:  # Use top 2 primary rules
            if rule in ["ColorReplacement", "FillHoles", "CropToBoundingBox", "ObjectCounting"]:
                recommended_chains.append(f"{rule} -> ReplaceBorderWithColor")
                recommended_chains.append(f"{rule} -> RemoveObjects")
            elif rule in ["DiagonalFlip", "RotateClockwise", "HorizontalFlip", "VerticalFlip"]:
                recommended_chains.append(f"{rule} -> ColorReplacement")
                recommended_chains.append(f"{rule} -> FillHoles")
            elif rule in ["TilePatternExpansion", "MirrorBandExpansion", "DuplicateRowsOrColumns"]:
                recommended_chains.append(f"{rule} -> ColorReplacement")
                recommended_chains.append(f"{rule} -> CropToBoundingBox")
    
    # Remove duplicates and return unique chains
    return list(dict.fromkeys(recommended_chains))
    top_transformations = []
    for pred in sorted(advanced_preprocessing['predictions'], 
                      key=lambda x: x.get('confidence', 0.0), 
                      reverse=True)[:2]:
        top_transformations.append(pred.get('type'))
    
    # Build recommended chains
    recommended_chains = []
    for trans_type in top_transformations:
        if trans_type in transformation_chains:
            recommended_chains.extend(transformation_chains[trans_type])
    
    return recommended_chains
