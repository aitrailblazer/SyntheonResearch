#!/usr/bin/env python3
"""
Enhanced Syntheon Main with Task Statistics Tracking
Integrates comprehensive task-level and example-level success tracking
"""

import os, copy, logging, subprocess
import xml.etree.ElementTree as ET
import numpy as np
from syntheon_engine import SyntheonEngine
from task_statistics import TaskStatisticsTracker, load_task_metadata_from_xml
from itertools import product
from collections import Counter
import math
import json
from datetime import datetime

# Clear the log file at the start of each run
with open("syntheon_output.log", "w") as f:
    pass

logging.basicConfig(
    filename="syntheon_output.log",
    level=logging.INFO,
    format="%(message)s"
)

def extract_kwic_features(input_element):
    """
    Extract KWIC (color co-occurrence) features from an input element.
    Returns a dictionary with color pattern statistics for rule prioritization.
    """
    kwic_el = input_element.find('kwic')
    if kwic_el is None:
        return {}
    
    features = {
        'total_cells': int(kwic_el.get('total_cells', 0)),
        'grid_count': int(kwic_el.get('grid_count', 0)),
        'window_size': int(kwic_el.get('window_size', 2)),
        'color_pairs': [],
        'dominant_colors': [],
        'rare_colors': [],
        'pattern_complexity': 0
    }
    
    # Extract color pair frequencies
    pair_frequencies = []
    for pair_el in kwic_el.findall('pair'):
        color1 = int(pair_el.get('color1'))
        color2 = int(pair_el.get('color2'))
        count = int(pair_el.get('count'))
        frequency = float(pair_el.get('frequency'))
        
        features['color_pairs'].append({
            'colors': (color1, color2),
            'count': count,
            'frequency': frequency
        })
        pair_frequencies.append(frequency)
    
    if pair_frequencies:
        # Calculate pattern complexity based on frequency distribution
        entropy = -sum(f * math.log2(f) for f in pair_frequencies if f > 0)
        features['pattern_complexity'] = entropy
        
        # Identify dominant and rare colors
        color_freq = Counter()
        for pair in features['color_pairs']:
            color_freq[pair['colors'][0]] += pair['frequency']
            color_freq[pair['colors'][1]] += pair['frequency']
        
        sorted_colors = sorted(color_freq.items(), key=lambda x: x[1], reverse=True)
        total_colors = len(sorted_colors)
        
        if total_colors > 0:
            features['dominant_colors'] = [c for c, f in sorted_colors[:max(1, total_colors//3)]]
            features['rare_colors'] = [c for c, f in sorted_colors[-max(1, total_colors//3):]]
    
    return features

def prioritize_rules_by_kwic(kwic_features, available_rules_meta):
    """
    Prioritize rules based on KWIC features and rule characteristics.
    Enhanced version with better pattern matching.
    """
    if not kwic_features or not available_rules_meta:
        return list(available_rules_meta.keys())
    
    rule_scores = {}
    pattern_complexity = kwic_features.get('pattern_complexity', 0)
    dominant_colors = set(kwic_features.get('dominant_colors', []))
    color_pairs = kwic_features.get('color_pairs', [])
    
    for rule_name, rule_meta in available_rules_meta.items():
        score = 0.5  # Base score
        
        # Pattern complexity matching
        if 'complexity' in rule_meta:
            rule_complexity = rule_meta['complexity']
            complexity_match = 1.0 - abs(pattern_complexity - rule_complexity) / max(pattern_complexity, rule_complexity, 1.0)
            score += 0.3 * complexity_match
        
        # Color sensitivity matching
        if 'color_sensitive' in rule_meta and rule_meta['color_sensitive']:
            if len(dominant_colors) > 1:
                score += 0.2
        
        # Pattern type matching
        rule_type = rule_meta.get('type', '')
        if pattern_complexity > 2.0 and 'pattern' in rule_type.lower():
            score += 0.3
        elif pattern_complexity < 1.0 and ('simple' in rule_type.lower() or 'basic' in rule_type.lower()):
            score += 0.3
        
        # Frequency pattern analysis
        if color_pairs:
            high_freq_pairs = [p for p in color_pairs if p['frequency'] > 0.3]
            if high_freq_pairs and 'repetition' in rule_type.lower():
                score += 0.2
        
        rule_scores[rule_name] = score
    
    # Sort rules by score (descending)
    prioritized_rules = sorted(rule_scores.items(), key=lambda x: x[1], reverse=True)
    return [rule_name for rule_name, score in prioritized_rules]

def choose_rule_with_params(rule_name, params_space, input_grid, target_grid):
    """Choose the best parameters for a rule by trying all combinations."""
    best_params = None
    best_similarity = -1
    
    # Generate all parameter combinations
    param_keys = list(params_space.keys())
    param_values = [params_space[key] for key in param_keys]
    
    for param_combination in product(*param_values):
        params = dict(zip(param_keys, param_combination))
        
        try:
            # Create temporary engine instance to test this rule with these params
            temp_engine = SyntheonEngine()
            temp_engine.load_rules_from_xml("syntheon_rules_glyphs.xml")
            
            if hasattr(temp_engine, rule_name):
                rule_func = getattr(temp_engine, rule_name)
                result_grid = rule_func(input_grid, **params)
                
                if result_grid is not None:
                    similarity = calculate_similarity(result_grid, target_grid)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_params = params
                        
        except Exception as e:
            continue
    
    return best_params, best_similarity

def calculate_similarity(grid1, grid2):
    """Calculate similarity between two grids."""
    if grid1 is None or grid2 is None:
        return 0.0
    
    g1 = np.array(grid1)
    g2 = np.array(grid2)
    
    if g1.shape != g2.shape:
        return 0.0
    
    return np.mean(g1 == g2)

def choose_rule_chain(input_grid, target_grid, engine, max_chain_length=3):
    """
    Choose the best rule chain by comprehensive parameter sweeping.
    Enhanced with KWIC prioritization and better parameter exploration.
    """
    best_chain = None
    best_similarity = -1
    best_final_grid = None
    
    # Extract KWIC features for prioritization
    # Note: In a full implementation, you'd extract these from the XML
    # For now, we'll use the available rules without KWIC prioritization
    available_rules = list(engine.rules_meta.keys())
    
    # Comprehensive parameter spaces for each rule
    parameter_spaces = {
        'TilePatternExpansion': {
            'target_width': [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
            'target_height': [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
        },
        'ColorReplacement': {
            'old_color': list(range(10)),
            'new_color': list(range(10))
        },
        'MirrorBandExpansion': {
            'band_width': [1, 2, 3, 4, 5],
            'direction': ['horizontal', 'vertical']
        },
        'DiagonalFlip': {},
        'RotateGrid': {
            'angle': [90, 180, 270]
        },
        'FillHoles': {
            'target_color': list(range(10))
        }
    }
    
    # Single rule attempts with comprehensive parameter sweeping
    for rule_name in available_rules:
        if rule_name in parameter_spaces:
            params_space = parameter_spaces[rule_name]
            best_params, similarity = choose_rule_with_params(rule_name, params_space, input_grid, target_grid)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_chain = [(rule_name, best_params)]
                
                # Apply the rule to get the final grid
                try:
                    if hasattr(engine, rule_name):
                        rule_func = getattr(engine, rule_name)
                        best_final_grid = rule_func(input_grid, **best_params) if best_params else rule_func(input_grid)
                except:
                    pass
    
    # Chain attempts (2-3 rules)
    for chain_length in range(2, max_chain_length + 1):
        for rule_combination in product(available_rules, repeat=chain_length):
            try:
                current_grid = copy.deepcopy(input_grid)
                chain_params = []
                
                for rule_name in rule_combination:
                    if rule_name in parameter_spaces:
                        params_space = parameter_spaces[rule_name]
                        # For chains, use simpler parameter exploration
                        best_params = None
                        best_step_similarity = -1
                        
                        for param_combination in product(*[params_space[key] for key in params_space.keys()]):
                            params = dict(zip(params_space.keys(), param_combination))
                            
                            try:
                                if hasattr(engine, rule_name):
                                    rule_func = getattr(engine, rule_name)
                                    temp_grid = rule_func(current_grid, **params)
                                    
                                    if temp_grid is not None:
                                        similarity = calculate_similarity(temp_grid, target_grid)
                                        if similarity > best_step_similarity:
                                            best_step_similarity = similarity
                                            best_params = params
                            except:
                                continue
                        
                        if best_params and hasattr(engine, rule_name):
                            rule_func = getattr(engine, rule_name)
                            current_grid = rule_func(current_grid, **best_params)
                            chain_params.append((rule_name, best_params))
                        else:
                            break
                    else:
                        # Rules without parameters
                        if hasattr(engine, rule_name):
                            rule_func = getattr(engine, rule_name)
                            current_grid = rule_func(current_grid)
                            chain_params.append((rule_name, {}))
                
                if current_grid is not None:
                    final_similarity = calculate_similarity(current_grid, target_grid)
                    if final_similarity > best_similarity:
                        best_similarity = final_similarity
                        best_chain = chain_params
                        best_final_grid = current_grid
                        
            except Exception as e:
                continue
    
    return best_chain, best_similarity, best_final_grid

def solve_single_example(input_grid, target_grid, engine):
    """
    Solve a single example using the engine.
    Returns (success, applied_rules, final_grid, similarity)
    """
    try:
        rule_chain, similarity, final_grid = choose_rule_chain(input_grid, target_grid, engine)
        
        success = similarity >= 0.99  # 99% match threshold
        applied_rules = [rule_name for rule_name, params in rule_chain] if rule_chain else []
        
        return success, applied_rules, final_grid, similarity
        
    except Exception as e:
        logging.error(f"Error solving example: {e}")
        return False, [], None, 0.0

def solve_task_with_statistics(task_element, engine, stats_tracker):
    """
    Solve a complete ARC task and record detailed statistics.
    """
    task_id = task_element.get('id')
    logging.info(f"\n{'='*60}")
    logging.info(f"Processing Task: {task_id}")
    logging.info(f"{'='*60}")
    
    # Extract training and test examples
    training_examples = task_element.find('training_examples')
    test_examples = task_element.find('test_examples')
    
    training_count = int(training_examples.get('count', 0)) if training_examples is not None else 0
    test_count = int(test_examples.get('count', 0)) if test_examples is not None else 0
    
    example_results = []
    successful_rules = set()
    
    # Process training examples
    if training_examples is not None:
        for example in training_examples.findall('example'):
            input_grid = parse_grid(example.find('input'))
            target_grid = parse_grid(example.find('output'))
            
            success, applied_rules, final_grid, similarity = solve_single_example(input_grid, target_grid, engine)
            example_results.append(success)
            
            if applied_rules:
                successful_rules.update(applied_rules)
            
            logging.info(f"Training Example {example.get('index')}: {'✓' if success else '✗'} (similarity: {similarity:.3f})")
            if applied_rules:
                logging.info(f"  Applied rules: {' → '.join(applied_rules)}")
    
    # Process test examples
    if test_examples is not None:
        for example in test_examples.findall('example'):
            input_grid = parse_grid(example.find('input'))
            # Note: For test examples, we might not have the target grid
            # This would depend on your XML structure
            output_element = example.find('output')
            if output_element is not None:
                target_grid = parse_grid(output_element)
                success, applied_rules, final_grid, similarity = solve_single_example(input_grid, target_grid, engine)
            else:
                # If no target grid, we can't evaluate success
                success = False
                applied_rules = []
                similarity = 0.0
            
            example_results.append(success)
            
            if applied_rules:
                successful_rules.update(applied_rules)
            
            logging.info(f"Test Example {example.get('index')}: {'✓' if success else '✗'} (similarity: {similarity:.3f})")
            if applied_rules:
                logging.info(f"  Applied rules: {' → '.join(applied_rules)}")
    
    # Record statistics
    stats_tracker.add_task_result(
        task_id=task_id,
        example_results=example_results,
        training_count=training_count,
        test_count=test_count,
        successful_rules=list(successful_rules)
    )
    
    solved_examples = sum(example_results)
    total_examples = len(example_results)
    task_success_rate = solved_examples / total_examples if total_examples > 0 else 0.0
    
    logging.info(f"\nTask {task_id} Summary:")
    logging.info(f"  Examples solved: {solved_examples}/{total_examples} ({task_success_rate:.1%})")
    logging.info(f"  Rules used: {', '.join(successful_rules) if successful_rules else 'None'}")
    logging.info(f"  Status: {'✓ FULLY SOLVED' if task_success_rate == 1.0 else '◐ PARTIALLY SOLVED' if task_success_rate > 0 else '✗ UNSOLVED'}")
    
    return task_success_rate

def parse_grid(grid_element):
    """Parse a grid element into a 2D numpy array."""
    if grid_element is None:
        return None
    
    height = int(grid_element.get('height'))
    width = int(grid_element.get('width'))
    
    grid = []
    for row_element in grid_element.findall('row'):
        row_text = row_element.text.strip()
        row = [int(x) for x in row_text.split()]
        grid.append(row)
    
    return np.array(grid)

def main():
    """Enhanced main function with comprehensive statistics tracking."""
    print("🚀 Starting Syntheon Enhanced with Task Statistics...")
    
    # Initialize engine and statistics tracker
    engine = SyntheonEngine()
    engine.load_rules_from_xml("syntheon_rules_glyphs.xml")
    stats_tracker = TaskStatisticsTracker()
    
    print(f"✓ Loaded {len(engine.rules_meta)} rules")
    
    # Load and process tasks
    xml_file = "input/arc_agi2_training_combined.xml"
    
    if not os.path.exists(xml_file):
        print(f"❌ XML file not found: {xml_file}")
        return
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    tasks = root.findall('arc_agi_task')
    total_tasks = len(tasks)
    
    print(f"📊 Processing {total_tasks} tasks...")
    
    # Process each task
    processed_tasks = 0
    for task_element in tasks:
        try:
            task_success_rate = solve_task_with_statistics(task_element, engine, stats_tracker)
            processed_tasks += 1
            
            # Print progress
            if processed_tasks % 50 == 0:
                current_stats = stats_tracker.compute_overall_statistics()
                print(f"Progress: {processed_tasks}/{total_tasks} tasks processed")
                print(f"Current stats: {current_stats.solved_examples}/{current_stats.total_examples} examples solved ({current_stats.overall_example_success_rate:.2%})")
                print(f"Fully solved tasks: {current_stats.fully_solved_tasks}/{current_stats.total_tasks} ({current_stats.full_task_success_rate:.2%})")
                
        except Exception as e:
            logging.error(f"Error processing task {task_element.get('id')}: {e}")
            continue
    
    # Generate final statistics
    print("\n" + "="*80)
    print("🎯 FINAL RESULTS")
    print("="*80)
    
    stats_tracker.print_summary()
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"task_results_{timestamp}.json"
    stats_tracker.save_results(results_file)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    print("🔍 Check syntheon_output.log for detailed execution logs")

if __name__ == "__main__":
    main()
