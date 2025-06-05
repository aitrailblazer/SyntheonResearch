#!/usr/bin/env python3
# hybrid_engine_adapter.py
"""
Adapter module that integrates the performance-optimized SyntheonEngine
with the glyph-based DSL interpreter for hybrid functionality using syntheon_rules_glyphs.xml.
"""

import numpy as np
import logging
import time
from syntheon_engine import SyntheonEngine
from glyph_interpreter import GlyphInterpreter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hybrid_engine')

class HybridEngine:
    """
    Hybrid engine combining SyntheonEngine and GlyphInterpreter, using rules from syntheon_rules_glyphs.xml.
    """
    
    def __init__(self, glyph_rules_xml_path, default_mode="dsl"):
        """
        Initialize the hybrid engine with glyph-based rules.
        
        Args:
            glyph_rules_xml_path: Path to syntheon_rules_glyphs.xml
            default_mode: Execution mode ('performance', 'dsl', or 'auto')
        """
        try:
            self.perf_engine = SyntheonEngine()
            self.perf_engine.load_rules_from_xml(glyph_rules_xml_path)
        except Exception as e:
            logger.warning(f"Failed to initialize SyntheonEngine: {e}, relying on DSL mode")
            self.perf_engine = None
        
        try:
            self.dsl_engine = GlyphInterpreter(glyph_rules_xml_path)
        except Exception as e:
            logger.error(f"Failed to initialize GlyphInterpreter: {e}")
            raise
        
        if not self.dsl_engine and not self.perf_engine:
            raise ValueError("At least one engine must be initialized")
        
        self.default_mode = default_mode if default_mode in ["performance", "dsl", "auto"] else "dsl"
        self.performance_stats = {}
        self.execution_history = []
        
        # Performance thresholds for mode switching
        self.dsl_slowdown_threshold = 1.5
        self.grid_size_threshold = 10
        
        logger.info(f"Hybrid engine initialized with default mode: {self.default_mode}")

    @property
    def rules_meta(self):
        """Expose rules_meta from DSL engine, falling back to performance engine."""
        if self.dsl_engine:
            return {name: {} for name in self.dsl_engine.rules}
        elif self.perf_engine:
            return self.perf_engine.rules_meta
        return {}

    def set_mode(self, mode):
        """Set the execution mode."""
        if mode not in ["performance", "dsl", "auto"]:
            raise ValueError(f"Invalid mode: {mode}")
        self.default_mode = mode
        logger.info(f"Execution mode set to: {mode}")
    
    def apply_rule(self, rule_name, grid, mode=None, **params):
        """
        Apply a rule to a grid using the specified or default execution mode.
        """
        if grid.size == 0:
            logger.warning("Empty grid provided, returning empty grid")
            return grid
        
        execution_mode = mode or self.default_mode
        if execution_mode == "auto":
            execution_mode = self._select_optimal_mode(rule_name, grid, params)
        
        start_time = time.time()
        result = None
        error = None
        
        try:
            if execution_mode == "dsl" and self.dsl_engine:
                result = self.dsl_engine.apply_rule(rule_name, grid, **params)
                execution_mode = "dsl"
            elif self.perf_engine:
                result = self.perf_engine.apply_rule(rule_name, grid, **params)
                execution_mode = "performance"
            else:
                raise ValueError("No valid engine available")
        except Exception as e:
            error = e
            logger.warning(f"Error in {execution_mode} mode: {e}")
            
            try:
                if execution_mode == "dsl" and self.perf_engine:
                    logger.info(f"Falling back to performance mode for {rule_name}")
                    result = self.perf_engine.apply_rule(rule_name, grid, **params)
                    execution_mode = "performance_fallback"
                elif execution_mode == "performance" and self.dsl_engine:
                    logger.info(f"Falling back to DSL mode for {rule_name}")
                    result = self.dsl_engine.apply_rule(rule_name, grid, **params)
                    execution_mode = "dsl_fallback"
                else:
                    raise
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise fallback_error
        
        execution_time = time.time() - start_time
        self._record_execution_stats(rule_name, execution_mode, grid.shape, execution_time, error)
        
        return result
    
    def _select_optimal_mode(self, rule_name, grid, params):
        """Select optimal execution mode."""
        if self.dsl_engine and rule_name in self.dsl_engine.rules:
            return "dsl"
        
        if max(grid.shape) > self.grid_size_threshold:
            return "performance" if self.perf_engine else "dsl"
        
        rule_stats = self.performance_stats.get(rule_name, {})
        if rule_stats:
            perf_avg = rule_stats.get("performance", {}).get("avg_time", 0)
            dsl_avg = rule_stats.get("dsl", {}).get("avg_time", 0)
            if perf_avg > 0 and dsl_avg > 0:
                ratio = dsl_avg / perf_avg
                if ratio > self.dsl_slowdown_threshold:
                    return "performance" if self.perf_engine else "dsl"
                return "dsl" if self.dsl_engine else "performance"
        
        return "dsl" if self.dsl_engine and grid.size < 25 else "performance" if self.perf_engine else "dsl"
    
    def _record_execution_stats(self, rule_name, mode, grid_shape, execution_time, error):
        """Record execution statistics."""
        if rule_name not in self.performance_stats:
            self.performance_stats[rule_name] = {
                "performance": {"count": 0, "total_time": 0, "avg_time": 0, "errors": 0},
                "dsl": {"count": 0, "total_time": 0, "avg_time": 0, "errors": 0}
            }
        
        stats_key = "dsl" if mode in ["dsl", "dsl_fallback"] else "performance"
        stats = self.performance_stats[rule_name][stats_key]
        stats["count"] += 1
        stats["total_time"] += execution_time
        stats["avg_time"] = stats["total_time"] / stats["count"]
        if error:
            stats["errors"] += 1
        
        self.execution_history.append({
            "rule": rule_name,
            "mode": mode,
            "grid_shape": grid_shape,
            "execution_time": execution_time,
            "error": str(error) if error else None,
            "timestamp": time.time()
        })
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    def get_performance_report(self):
        """Generate a performance comparison report."""
        report = {
            "rule_performance": {},
            "mode_summary": {
                "performance": {"count": 0, "total_time": 0, "avg_time": 0, "errors": 0},
                "dsl": {"count": 0, "total_time": 0, "avg_time": 0, "errors": 0}
            }
        }
        
        for rule_name, stats in self.performance_stats.items():
            perf_stats = stats["performance"]
            dsl_stats = stats["dsl"]
            ratio = dsl_stats["avg_time"] / perf_stats["avg_time"] if perf_stats["count"] > 0 and dsl_stats["count"] > 0 else None
            
            report["rule_performance"][rule_name] = {
                "performance": {
                    "count": perf_stats["count"],
                    "avg_time": perf_stats["avg_time"],
                    "errors": perf_stats["errors"]
                },
                "dsl": {
                    "count": dsl_stats["count"],
                    "avg_time": dsl_stats["avg_time"],
                    "errors": dsl_stats["errors"]
                },
                "ratio": ratio
            }
            
            for mode in ["performance", "dsl"]:
                mode_stats = stats[mode]
                summary = report["mode_summary"][mode]
                summary["count"] += mode_stats["count"]
                summary["total_time"] += mode_stats["total_time"]
                summary["errors"] += mode_stats["errors"]
                if summary["count"] > 0:
                    summary["avg_time"] = summary["total_time"] / summary["count"]
        
        perf_summary = report["mode_summary"]["performance"]
        dsl_summary = report["mode_summary"]["dsl"]
        report["overall_ratio"] = dsl_summary["avg_time"] / perf_summary["avg_time"] if perf_summary["count"] > 0 and dsl_summary["count"] > 0 else None
        
        return report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid engine demonstration")
    parser.add_argument("--glyph-rules", default="syntheon_rules_glyphs.xml", help="Path to glyph rules XML")
    parser.add_argument("--mode", default="dsl", choices=["performance", "dsl", "auto"], help="Execution mode")
    args = parser.parse_args()
    
    engine = HybridEngine(args.glyph_rules, args.mode)
    
    grid = np.array([[1, 2], [3, 4]])
    # print(f"Running TilePatternExpansion on {grid} in {args.mode} mode:")
    try:
        result = engine.apply_rule("TilePatternExpansion", grid)
        # print(result)
    except Exception as e:
        # print(f"Error: {e}")
    
    # print("\nRunning ColorReplacement on larger grid:")
    large_grid = np.random.randint(0, 4, size=(10, 10))
    try:
        result = engine.apply_rule("ColorReplacement", large_grid, from_color=1, to_color=5)
        # print(f"Result shape: {result.shape}")
    except Exception as e:
        # print(f"Error: {e}")
    
    # print("\nPerformance Report:")
    report = engine.get_performance_report()
    for rule, stats in report["rule_performance"].items():
        # print(f"Rule: {rule}")
        if stats["ratio"]:
            # print(f"  Performance ratio (DSL/Performance): {stats['ratio']:.2f}x")
        # print(f"  Performance mode: {stats['performance']['count']} executions, {stats['performance']['avg_time']*1000:.2f}ms avg")
        # print(f"  DSL mode: {stats['dsl']['count']} executions, {stats['dsl']['avg_time']*1000:.2f}ms avg")
        # print()
    if report["overall_ratio"]:
        # print(f"Overall performance ratio (DSL/Performance): {report['overall_ratio']:.2f}x")