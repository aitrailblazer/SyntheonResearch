#!/usr/bin/env python3
"""
ARC Task Statistics Tracker
Tracks full task success, partial task success, and individual example success
"""

import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class TaskResult:
    task_id: str
    total_examples: int
    solved_examples: int
    test_examples: int
    solved_test_examples: int
    training_examples: int
    solved_training_examples: int
    success_rate: float
    is_fully_solved: bool
    is_partially_solved: bool
    solved_example_indices: List[int]
    failed_example_indices: List[int]

@dataclass
class OverallStatistics:
    total_tasks: int
    fully_solved_tasks: int
    partially_solved_tasks: int
    unsolved_tasks: int
    total_examples: int
    solved_examples: int
    
    # Task-level statistics
    full_task_success_rate: float
    partial_task_success_rate: float
    
    # Example-level statistics  
    overall_example_success_rate: float
    
    # Training vs Test breakdown
    total_training_examples: int
    solved_training_examples: int
    total_test_examples: int
    solved_test_examples: int
    training_success_rate: float
    test_success_rate: float

class TaskStatisticsTracker:
    def __init__(self):
        self.task_results: Dict[str, TaskResult] = {}
        self.rule_success_by_task: Dict[str, List[str]] = {}
        
    def _json_serializer(self, obj):
        """Custom JSON serializer to handle numpy types and dataclasses"""
        import numpy as np
        from dataclasses import asdict, is_dataclass
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif is_dataclass(obj):
            return asdict(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
    def add_task_result(self, task_id: str, example_results: List[bool], 
                       training_count: int, test_count: int, 
                       successful_rules: List[str] = None):
        """
        Add results for a specific task
        
        Args:
            task_id: Unique task identifier
            example_results: List of boolean results for each example (training + test)
            training_count: Number of training examples
            test_count: Number of test examples  
            successful_rules: List of rules that succeeded for this task
        """
        total_examples = len(example_results)
        solved_examples = sum(example_results)
        
        # Split training and test results
        training_results = example_results[:training_count]
        test_results = example_results[training_count:]
        
        solved_training = sum(training_results)
        solved_test = sum(test_results)
        
        success_rate = solved_examples / total_examples if total_examples > 0 else 0.0
        is_fully_solved = solved_examples == total_examples
        is_partially_solved = solved_examples > 0 and not is_fully_solved
        
        solved_indices = [i for i, result in enumerate(example_results) if result]
        failed_indices = [i for i, result in enumerate(example_results) if not result]
        
        task_result = TaskResult(
            task_id=task_id,
            total_examples=total_examples,
            solved_examples=solved_examples,
            test_examples=test_count,
            solved_test_examples=solved_test,
            training_examples=training_count,
            solved_training_examples=solved_training,
            success_rate=success_rate,
            is_fully_solved=is_fully_solved,
            is_partially_solved=is_partially_solved,
            solved_example_indices=solved_indices,
            failed_example_indices=failed_indices
        )
        
        self.task_results[task_id] = task_result
        
        if successful_rules:
            self.rule_success_by_task[task_id] = successful_rules
    
    def compute_overall_statistics(self) -> OverallStatistics:
        """Compute comprehensive statistics across all tasks"""
        if not self.task_results:
            return OverallStatistics(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.0, 0.0)
        
        total_tasks = len(self.task_results)
        fully_solved_tasks = sum(1 for task in self.task_results.values() if task.is_fully_solved)
        partially_solved_tasks = sum(1 for task in self.task_results.values() if task.is_partially_solved)
        unsolved_tasks = total_tasks - fully_solved_tasks - partially_solved_tasks
        
        total_examples = sum(task.total_examples for task in self.task_results.values())
        solved_examples = sum(task.solved_examples for task in self.task_results.values())
        
        total_training = sum(task.training_examples for task in self.task_results.values())
        solved_training = sum(task.solved_training_examples for task in self.task_results.values())
        total_test = sum(task.test_examples for task in self.task_results.values())
        solved_test = sum(task.solved_test_examples for task in self.task_results.values())
        
        return OverallStatistics(
            total_tasks=total_tasks,
            fully_solved_tasks=fully_solved_tasks,
            partially_solved_tasks=partially_solved_tasks,
            unsolved_tasks=unsolved_tasks,
            total_examples=total_examples,
            solved_examples=solved_examples,
            full_task_success_rate=fully_solved_tasks / total_tasks,
            partial_task_success_rate=(fully_solved_tasks + partially_solved_tasks) / total_tasks,
            overall_example_success_rate=solved_examples / total_examples if total_examples > 0 else 0.0,
            total_training_examples=total_training,
            solved_training_examples=solved_training,
            total_test_examples=total_test,
            solved_test_examples=solved_test,
            training_success_rate=solved_training / total_training if total_training > 0 else 0.0,
            test_success_rate=solved_test / total_test if total_test > 0 else 0.0
        )
    
    def get_detailed_breakdown(self) -> Dict:
        """Get detailed breakdown of results"""
        stats = self.compute_overall_statistics()
        
        # Group tasks by success level
        fully_solved = [task_id for task_id, task in self.task_results.items() if task.is_fully_solved]
        partially_solved = [task_id for task_id, task in self.task_results.items() if task.is_partially_solved]
        unsolved = [task_id for task_id, task in self.task_results.items() 
                   if not task.is_fully_solved and not task.is_partially_solved]
        
        # Success rate distribution
        success_rates = [task.success_rate for task in self.task_results.values()]
        success_rate_bins = {
            "100%": len([r for r in success_rates if r == 1.0]),
            "75-99%": len([r for r in success_rates if 0.75 <= r < 1.0]),
            "50-74%": len([r for r in success_rates if 0.50 <= r < 0.75]),
            "25-49%": len([r for r in success_rates if 0.25 <= r < 0.50]),
            "1-24%": len([r for r in success_rates if 0.01 <= r < 0.25]),
            "0%": len([r for r in success_rates if r == 0.0])
        }
        
        return {
            "overall_statistics": stats,
            "task_breakdown": {
                "fully_solved": {
                    "count": len(fully_solved),
                    "task_ids": fully_solved
                },
                "partially_solved": {
                    "count": len(partially_solved),
                    "task_ids": partially_solved
                },
                "unsolved": {
                    "count": len(unsolved),
                    "task_ids": unsolved
                }
            },
            "success_rate_distribution": success_rate_bins,
            "top_performing_tasks": self._get_top_performing_tasks(10),
            "rule_effectiveness": self._analyze_rule_effectiveness()
        }
    
    def _get_top_performing_tasks(self, limit: int) -> List[Dict]:
        """Get top performing tasks by success rate"""
        sorted_tasks = sorted(
            self.task_results.items(),
            key=lambda x: (x[1].success_rate, x[1].solved_examples),
            reverse=True
        )
        
        return [
            {
                "task_id": task_id,
                "success_rate": task.success_rate,
                "solved_examples": f"{task.solved_examples}/{task.total_examples}",
                "rules_used": self.rule_success_by_task.get(task_id, [])
            }
            for task_id, task in sorted_tasks[:limit]
        ]
    
    def _analyze_rule_effectiveness(self) -> Dict[str, Dict]:
        """Analyze which rules are most effective"""
        rule_stats = defaultdict(lambda: {"tasks_used": 0, "total_success_rate": 0.0})
        
        for task_id, rules in self.rule_success_by_task.items():
            task_success_rate = self.task_results[task_id].success_rate
            
            for rule in rules:
                rule_stats[rule]["tasks_used"] += 1
                rule_stats[rule]["total_success_rate"] += task_success_rate
        
        # Calculate average success rate per rule
        for rule, stats in rule_stats.items():
            if stats["tasks_used"] > 0:
                stats["average_success_rate"] = stats["total_success_rate"] / stats["tasks_used"]
            else:
                stats["average_success_rate"] = 0.0
        
        # Sort by effectiveness
        sorted_rules = sorted(
            rule_stats.items(),
            key=lambda x: (x[1]["average_success_rate"], x[1]["tasks_used"]),
            reverse=True
        )
        
        return dict(sorted_rules)
    
    def print_summary(self):
        """Print a comprehensive summary of results"""
        stats = self.compute_overall_statistics()
        breakdown = self.get_detailed_breakdown()
        
        print("=" * 80)
        print("ARC TASK STATISTICS SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"  Total Tasks: {stats.total_tasks}")
        print(f"  Fully Solved Tasks: {stats.fully_solved_tasks} ({stats.full_task_success_rate:.2%})")
        print(f"  Partially Solved Tasks: {stats.partially_solved_tasks} ({(stats.partially_solved_tasks/stats.total_tasks):.2%})")
        print(f"  Unsolved Tasks: {stats.unsolved_tasks} ({(stats.unsolved_tasks/stats.total_tasks):.2%})")
        
        print(f"\n🎯 EXAMPLE-LEVEL PERFORMANCE:")
        print(f"  Total Examples: {stats.total_examples}")
        print(f"  Solved Examples: {stats.solved_examples} ({stats.overall_example_success_rate:.2%})")
        print(f"  Training Examples: {stats.solved_training_examples}/{stats.total_training_examples} ({stats.training_success_rate:.2%})")
        print(f"  Test Examples: {stats.solved_test_examples}/{stats.total_test_examples} ({stats.test_success_rate:.2%})")
        
        print(f"\n📈 SUCCESS RATE DISTRIBUTION:")
        for rate_range, count in breakdown["success_rate_distribution"].items():
            percentage = count / stats.total_tasks * 100 if stats.total_tasks > 0 else 0
            print(f"  {rate_range}: {count} tasks ({percentage:.1f}%)")
        
        print(f"\n🏆 TOP PERFORMING TASKS:")
        for i, task_info in enumerate(breakdown["top_performing_tasks"][:5], 1):
            print(f"  {i}. {task_info['task_id']}: {task_info['success_rate']:.1%} ({task_info['solved_examples']})")
        
        print(f"\n🔧 MOST EFFECTIVE RULES:")
        rule_effectiveness = breakdown["rule_effectiveness"]
        for i, (rule, stats) in enumerate(list(rule_effectiveness.items())[:5], 1):
            print(f"  {i}. {rule}: {stats['average_success_rate']:.1%} avg success, used in {stats['tasks_used']} tasks")
    
    def save_results(self, filename: str):
        """Save results to JSON file"""
        results = {
            "overall_statistics": self.compute_overall_statistics().__dict__,
            "detailed_breakdown": self.get_detailed_breakdown(),
            "task_results": {
                task_id: {
                    "task_id": task.task_id,
                    "total_examples": task.total_examples,
                    "solved_examples": task.solved_examples,
                    "success_rate": task.success_rate,
                    "is_fully_solved": task.is_fully_solved,
                    "is_partially_solved": task.is_partially_solved,
                    "solved_example_indices": task.solved_example_indices,
                    "failed_example_indices": task.failed_example_indices,
                    "training_examples": task.training_examples,
                    "solved_training_examples": task.solved_training_examples,
                    "test_examples": task.test_examples,
                    "solved_test_examples": task.solved_test_examples
                }
                for task_id, task in self.task_results.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=self._json_serializer)
        
        print(f"Results saved to {filename}")

def load_task_metadata_from_xml(xml_file: str) -> Dict[str, Tuple[int, int]]:
    """
    Load task metadata (training and test example counts) from XML file
    
    Returns:
        Dict mapping task_id to (training_count, test_count)
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    task_metadata = {}
    
    for task_elem in root.findall('arc_agi_task'):
        task_id = task_elem.get('id')
        
        training_examples = task_elem.find('training_examples')
        test_examples = task_elem.find('test_examples')
        
        training_count = int(training_examples.get('count', 0)) if training_examples is not None else 0
        test_count = int(test_examples.get('count', 0)) if test_examples is not None else 0
        
        task_metadata[task_id] = (training_count, test_count)
    
    return task_metadata

if __name__ == "__main__":
    # Example usage
    tracker = TaskStatisticsTracker()
    
    # Load task metadata
    xml_file = "input/arc_agi2_training_combined.xml"
    task_metadata = load_task_metadata_from_xml(xml_file)
    
    print(f"Loaded metadata for {len(task_metadata)} tasks")
    
    # Example: Add some sample results
    for task_id, (training_count, test_count) in list(task_metadata.items())[:5]:
        # Simulate some results (in reality these would come from running Syntheon)
        total_examples = training_count + test_count
        # Simulate varying success rates
        import random
        success_rate = random.random()
        solved_count = int(success_rate * total_examples)
        
        example_results = [True] * solved_count + [False] * (total_examples - solved_count)
        random.shuffle(example_results)
        
        successful_rules = ["TilePatternExpansion", "ColorReplacement"] if solved_count > 0 else []
        
        tracker.add_task_result(task_id, example_results, training_count, test_count, successful_rules)
    
    tracker.print_summary()
