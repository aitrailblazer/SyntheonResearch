# glyph_interpreter.py
import numpy as np
import xml.etree.ElementTree as ET
import re
import logging
from scipy.ndimage import label, binary_fill_holes
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("glyph_interpreter")

# ---- GLYPH → PRIMITIVE MAPPING ----

def tile_pattern_expansion(grid, **kwargs):
    if grid.shape != (2, 2):
        raise ValueError(f"Expected 2x2 tile, got {grid.shape}")
    return np.tile(grid, (3, 3))

def mirror_band_expansion(grid, **kwargs):
    return np.array([np.concatenate([row, row[::-1]]) for row in grid], dtype=grid.dtype)

def frame_fill_convergence(grid, **kwargs):
    h, w = grid.shape
    out = np.ones((h, w), dtype=grid.dtype)
    out[:] = 1
    if h > 0 and w > 0:
        out[0, :] = 3
        out[-1, :] = 3
        out[:, 0] = 3
        out[:, -1] = 3
    return out

def color_replacement(grid, from_color=0, to_color=1, **kwargs):
    out = grid.copy()
    out[out == from_color] = to_color
    return out

def majority_fill(grid, **kwargs):
    vals, counts = np.unique(grid[grid != 0], return_counts=True)
    if len(vals) == 0:
        # logger.warning("No non-zero values for majority fill, returning original grid")
        return grid.copy()
    majority = vals[np.argmax(counts)]
    out = grid.copy()
    out[out == 0] = majority
    return out

def diagonal_flip(grid, **kwargs):
    return grid.T

def crop_to_bounding_box(grid, **kwargs):
    mask = grid != 0
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        # logger.warning("No non-zero elements, returning original grid")
        return grid
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return grid[rmin:rmax+1, cmin:cmax+1]

def remove_objects(grid, color=1, **kwargs):
    mask = (grid == color)
    labeled, n = label(mask)
    out = grid.copy()
    for i in range(1, n+1):
        out[labeled == i] = 0
    return out

def duplicate_rows_or_columns(grid, axis=0, n=2, **kwargs):
    if axis not in [0, 1]:
        logger.error(f"Invalid axis {axis}, defaulting to 0")
        axis = 0
    if n < 1:
        logger.error(f"Invalid n {n}, defaulting to 2")
        n = 2
    return np.repeat(grid, n, axis=axis)

def replace_border_with_color(grid, color=1, **kwargs):
    out = grid.copy()
    h, w = grid.shape
    if h > 0 and w > 0:
        out[0, :] = color
        out[-1, :] = color
        out[:, 0] = color
        out[:, -1] = color
    return out

def fill_holes(grid, **kwargs):
    out = grid.copy()
    for val in np.unique(grid):
        if val == 0:
            continue
        mask = (grid == val)
        filled = binary_fill_holes(mask)
        out[(filled & ~mask)] = val
    return out

def object_counting(grid, **kwargs):
    mask = (grid != 0)
    labeled, n = label(mask)
    return np.array([[n]], dtype=grid.dtype)

def color_swapping(grid, color_a=0, color_b=1, **kwargs):
    out = grid.copy()
    a_mask = out == color_a
    b_mask = out == color_b
    out[a_mask] = color_b
    out[b_mask] = color_a
    return out

def sequential_rule_application(grid, rules=None, interpreter=None, **kwargs):
    if not rules:
        # logger.warning("No rules provided for SequentialRuleApplication, returning original grid")
        return grid
    if not interpreter:
        logger.error("Interpreter required for SequentialRuleApplication")
        raise ValueError("Interpreter not provided")
    g = grid
    for rule_name in rules:
        try:
            rule_params = kwargs.get(rule_name, {})
            g = interpreter.apply_rule(rule_name, g, **rule_params)
        except Exception as e:
            # logger.warning(f"Failed to apply rule {rule_name} in sequence: {e}")
            continue
    return g

def conditional_rule_switch(grid, condition=None, if_true=None, if_false=None, interpreter=None, **kwargs):
    if not condition or not if_true:
        # logger.warning("Missing condition or if_true for ConditionalRuleSwitch, returning original grid")
        return grid
    if not interpreter:
        logger.error("Interpreter required for ConditionalRuleSwitch")
        raise ValueError("Interpreter not provided")
    
    # Evaluate condition
    if condition == "is_square":
        is_true = grid.shape[0] == grid.shape[1]
    elif condition == "has_symmetry":
        is_true = np.array_equal(grid, grid[:, ::-1])
    elif condition == "is_empty":
        is_true = np.all(grid == 0)
    elif condition == "has_color":
        color = kwargs.get("color", 0)
        is_true = np.any(grid == color)
    elif condition == "is_single_color":
        unique_colors = np.unique(grid[grid != 0])
        is_true = len(unique_colors) == 1
    else:
        # logger.warning(f"Unknown condition '{condition}', defaulting to False")
        is_true = False
    
    # Apply appropriate rule
    target_rule = if_true if is_true else if_false
    if not target_rule:
        return grid
    try:
        rule_params = kwargs.get(target_rule, {})
        return interpreter.apply_rule(target_rule, grid, **rule_params)
    except Exception as e:
        # logger.warning(f"Failed to apply rule {target_rule}: {e}")
        return grid

# --- GLYPH TABLE ---
GLYPH_PRIMITIVES = {
    "✦": "color_replacement",
    "✕": "color_swapping",
    "⧖": "mirror_band_expansion",
    "⟡": "diagonal_flip",
    "⛬": "crop_to_bounding_box",
    "█": "replace_border_with_color",
    "◐": "duplicate_rows_or_columns",
    "🜄": "remove_objects",
    "◼": "fill_holes",
}

# --- NAMED PRIMITIVE TABLE ---
PRIMITIVES = {
    "TilePatternExpansion": tile_pattern_expansion,
    "MirrorBandExpansion": mirror_band_expansion,
    "FrameFillConvergence": frame_fill_convergence,
    "ColorReplacement": color_replacement,
    "MajorityFill": majority_fill,
    "DiagonalFlip": diagonal_flip,
    "CropToBoundingBox": crop_to_bounding_box,
    "RemoveObjects": remove_objects,
    "DuplicateRowsOrColumns": duplicate_rows_or_columns,
    "ReplaceBorderWithColor": replace_border_with_color,
    "FillHoles": fill_holes,
    "ObjectCounting": object_counting,
    "ColorSwapping": color_swapping,
    "SequentialRuleApplication": sequential_rule_application,
    "ConditionalRuleSwitch": conditional_rule_switch,
}

# --- PARSER/EXECUTOR ---

class GlyphRule:
    def __init__(self, name, glyph_chain, param_defs=None):
        self.name = name
        self.chain = self.parse_chain(glyph_chain)
        self.param_defs = param_defs or []

    def parse_chain(self, chain_string):
        chain_string = chain_string.strip()
        if "->" in chain_string or "if(" in chain_string:
            return [chain_string]
        glyphs = [c for c in chain_string if not c.isspace()]
        return glyphs

    def execute(self, grid, param_bindings, interpreter):
        if grid.size == 0:
            # logger.warning("Empty grid provided, returning empty grid")
            return grid
        
        g = grid
        if len(self.chain) == 1 and ("->" in str(self.chain[0]) or "if(" in str(self.chain[0])):
            return self._exec_meta_chain(self.chain[0], g, param_bindings, interpreter)
        
        if self.name in PRIMITIVES:
            fn = PRIMITIVES[self.name]
            params = {p: param_bindings[p] for p in self.param_defs if p in param_bindings}
            return fn(g, **params, interpreter=interpreter)
        
        for glyph in self.chain:
            pname = GLYPH_PRIMITIVES.get(glyph)
            if pname is None:
                logger.error(f"No primitive for glyph {glyph}")
                raise NotImplementedError(f"No primitive for glyph {glyph}")
            fn = globals().get(pname)
            params = {p: param_bindings[p] for p in self.param_defs if p in param_bindings}
            g = fn(g, **params)
        return g

    def _exec_meta_chain(self, meta, grid, param_bindings, interpreter):
        if "->" in meta:
            rules = re.findall(r"\[(\w+)\]", meta) or param_bindings.get("rules", [])
            if not rules:
                logger.error("No rules found in sequential chain")
                raise ValueError("No rules found in sequential chain")
            g = grid
            for r in rules:
                try:
                    rule_params = param_bindings.get(r, {})
                    g = interpreter.apply_rule(r, g, **rule_params)
                except Exception as e:
                    # logger.warning(f"Failed to apply rule {r} in sequence: {e}")
                    continue
            return g
        
        if meta.startswith("if("):
            pred_match = re.match(r"if\((.*?)\)\{\[(.*?)\]\}else\{\[(.*?)\]\}", meta)
            if not pred_match:
                logger.error("Malformed meta-chain conditional")
                raise ValueError("Malformed meta-chain conditional")
            pred = pred_match.group(1)
            rule_true = pred_match.group(2)
            rule_false = pred_match.group(3)

            if pred == "is_square":
                is_true = grid.shape[0] == grid.shape[1]
            elif pred == "has_symmetry":
                is_true = np.array_equal(grid, grid[:, ::-1])
            elif pred == "is_empty":
                is_true = np.all(grid == 0)
            elif pred == "has_color":
                color = param_bindings.get("color", 0)
                is_true = np.any(grid == color)
            elif pred == "is_single_color":
                unique_colors = np.unique(grid[grid != 0])
                is_true = len(unique_colors) == 1
            else:
                # logger.warning(f"Unknown predicate '{pred}', defaulting to False")
                is_true = False

            target_rule = rule_true if is_true else rule_false
            try:
                rule_params = param_bindings.get(target_rule, {})
                return interpreter.apply_rule(target_rule, grid, **rule_params)
            except Exception as e:
                # logger.warning(f"Failed to apply rule {target_rule}: {e}")
                return grid
        
        logger.error("Meta-chain format not recognized")
        raise NotImplementedError("Meta-chain format not recognized")

class GlyphInterpreter:
    def __init__(self, xml_path):
        self.rules = {}
        self._rule_cache = {}  # Cache for rule lookups
        self.load_rules(xml_path)

    def load_rules(self, xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for rule in root.findall('rule'):
                name = rule.get('name')
                glyph_chain = rule.findtext('glyph_chain', '').strip()
                if not glyph_chain:
                    # logger.warning(f"Skipping rule {name} with empty glyph chain")
                    continue
                params = []
                param_block = rule.find('parameters')
                if param_block is not None:
                    for p in param_block.findall('param'):
                        params.append(p.get('name'))
                self.rules[name] = GlyphRule(name, glyph_chain, params)
                self._rule_cache[name] = self.rules[name]
        except Exception as e:
            logger.error(f"Failed to load rules from {xml_path}: {e}")
            raise

    def list_rules(self):
        for name in self.rules:
            print(f"{name}: {self.rules[name].chain}")

    @lru_cache(maxsize=128)
    def _get_rule(self, rule_name):
        return self._rule_cache.get(rule_name)

    def apply_rule(self, rule_name, grid, **params):
        # # logger.info(f"Applying rule: {rule_name}")
        rule = self._get_rule(rule_name)
        if not rule:
            logger.error(f"No such rule: {rule_name}")
            raise ValueError(f"No such rule: {rule_name}")
        
        if grid.size == 0:
            # logger.warning("Empty grid provided, returning empty grid")
            return grid
        
        # Parameter inference
        if rule_name == "RemoveObjects" and "color" not in params:
            # logger.info("Inferring color for RemoveObjects")
            vals, counts = np.unique(grid[grid != 0], return_counts=True)
            params["color"] = vals[np.argmax(counts)] if len(vals) > 0 else 1
                
        elif rule_name == "ColorReplacement" and ("from_color" not in params or "to_color" not in params):
            # logger.info("Inferring from_color and to_color for ColorReplacement")
            unique_colors = np.unique(grid)
            params["from_color"] = params.get("from_color", unique_colors[0] if len(unique_colors) > 0 else 0)
            params["to_color"] = params.get("to_color", unique_colors[1] if len(unique_colors) > 1 else 1 if len(unique_colors) == 1 else 0)

        elif rule_name == "ConditionalRuleSwitch":
            # logger.info("Handling ConditionalRuleSwitch")
            condition = params.get("condition", "is_square")
            if_true = params.get("if_true")
            if_false = params.get("if_false")
            return conditional_rule_switch(grid, condition, if_true, if_false, interpreter=self, **params)
        
        elif rule_name == "SequentialRuleApplication":
            # logger.info("Handling SequentialRuleApplication")
            rules = params.get("rules", [])
            return sequential_rule_application(grid, rules, interpreter=self, **params)
            
        elif rule_name == "DuplicateRowsOrColumns" and ("axis" not in params or "n" not in params):
            # logger.info("Inferring axis and n for DuplicateRowsOrColumns")
            params["axis"] = 0 if grid.shape[0] < grid.shape[1] else 1
            params["n"] = 2
            
        elif rule_name == "ReplaceBorderWithColor" and "color" not in params:
            # logger.info("Inferring color for ReplaceBorderWithColor")
            unique_colors = np.unique(grid)
            params["color"] = unique_colors[-1] if len(unique_colors) > 1 else 1
                
        elif rule_name == "ColorSwapping" and ("color_a" not in params or "color_b" not in params):
            # logger.info("Inferring color_a and color_b for ColorSwapping")
            vals = np.unique(grid)
            params["color_a"] = vals[0] if len(vals) > 0 else 0
            params["color_b"] = vals[1] if len(vals) > 1 else 1 if len(vals) == 1 else 0
        
        try:
            return rule.execute(grid, params, self)
        except Exception as e:
            logger.error(f"Error executing rule {rule_name}: {e}")
            raise