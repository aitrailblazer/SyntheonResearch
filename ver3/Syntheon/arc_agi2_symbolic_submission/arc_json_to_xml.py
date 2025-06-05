import json
import xml.etree.ElementTree as ET
from collections import Counter
import os

# ---- Hardcoded file paths ----
TRAIN_CHALLENGES_JSON = "arc-prize-2025/arc-agi_training_challenges.json"
TRAIN_SOLUTIONS_JSON = "arc-prize-2025/arc-agi_training_solutions.json"
TEST_CHALLENGES_JSON = "arc-prize-2025/arc-agi_test_challenges.json"  # No solutions available
OUTPUT_XML           = "input/arc_agi2_training_combined.xml"
TEST_OUTPUT_XML      = "input/arc_agi2_test_combined.xml"

def parse_grid(grid):
    """
    Convert a 2D grid (list of lists) into shape, color stats, rows, and histogram.
    Returns a dictionary with height, width, area, colors, histogram, and text rows.
    """
    height = len(grid)
    width = len(grid[0]) if height else 0
    color_counts = Counter()
    for row in grid:
        color_counts.update(row)
    return {
        "height": height,
        "width": width,
        "area": height * width,
        "colors": list(color_counts.keys()),
        "histogram": dict(color_counts),
        "rows": [" ".join(map(str, row)) for row in grid]
    }

def annotate_color_roles(histogram):
    """
    Heuristically label color roles for the task.
    Returns a dict: {color: role}.
    - background: most frequent color
    - object: rare or unique colors
    - frame: edge-dominant color (if detected)
    """
    if not histogram:
        return {}
    bg_color = max(histogram, key=histogram.get)
    total_pixels = sum(histogram.values())
    rare_colors = [c for c, count in histogram.items() if count <= (0.05 * total_pixels)]
    color_roles = {str(bg_color): "background"}
    for c in rare_colors:
        color_roles[str(c)] = "object"
    # Optionally: detect frame color by edge frequency (not implemented here)
    return color_roles

def hypothesize_transformation(input_stats, output_stats):
    """
    Suggest candidate transformation types based on size/shape deltas.
    e.g., crop, tiling, downscale, color-filter, object-extract
    """
    ih, iw = input_stats["height"], input_stats["width"]
    oh, ow = output_stats["height"], output_stats["width"]
    hint = []
    if (ih == oh and iw == ow):
        hint.append("identity")
    elif (ih > oh or iw > ow):
        hint.append("crop_or_downscale")
    elif (ih < oh or iw < ow):
        hint.append("tiling_or_expand")
    if len(output_stats["colors"]) < len(input_stats["colors"]):
        hint.append("color-filter")
    if oh == 1 or ow == 1:
        hint.append("project_to_line")
    return ", ".join(hint)

def make_metadata_element(colors, grid_sizes, trans_list, all_histograms, transformation_hints):
    """
    Build a <metadata> element with color/role, grid sizes, transformations, and hints.
    """
    meta = ET.Element("metadata")
    statistics = ET.SubElement(meta, "statistics")
    ET.SubElement(statistics, "colors", count=str(len(colors))).text = " ".join(map(str, sorted(colors)))
    in_sizes_el = ET.SubElement(statistics, "input_sizes")
    for s in grid_sizes['inputs']:
        ET.SubElement(in_sizes_el, "size", height=str(s['height']), width=str(s['width']), area=str(s['area']))
    out_sizes_el = ET.SubElement(statistics, "output_sizes")
    for s in grid_sizes['outputs']:
        ET.SubElement(out_sizes_el, "size", height=str(s['height']), width=str(s['width']), area=str(s['area']))
    if trans_list:
        trans_el = ET.SubElement(statistics, "transformations")
        for i, tr in enumerate(trans_list):
            ET.SubElement(trans_el, "transformation", index=str(i)).text = tr
    # Color roles section
    roles_el = ET.SubElement(statistics, "color_roles")
    color_roles = {}
    for hist in all_histograms:
        for c, role in annotate_color_roles(hist).items():
            color_roles[c] = role
    for c, role in color_roles.items():
        ET.SubElement(roles_el, "role", color=str(c)).text = role
    # Transformation hints
    if transformation_hints:
        hints_el = ET.SubElement(statistics, "transformation_hypotheses")
        for idx, h in enumerate(transformation_hints):
            if h:
                ET.SubElement(hints_el, "hint", index=str(idx)).text = h
    return meta

def kwic_for_colors(grids, window=2):
    """
    Generate a KWIC-style (Key Word In Context) color co-occurrence map.
    Useful for structural analysis of color patterns in local neighborhoods.
    
    Args:
        grids: List of 2D grid arrays
        window: Neighborhood window size (default 2 = 5x5 window)
    
    Returns:
        XML element with color co-occurrence statistics
    """
    co_occur = Counter()
    total_cells = 0
    
    for grid in grids:
        h = len(grid)
        w = len(grid[0]) if h else 0
        total_cells += h * w
        
        for i in range(h):
            for j in range(w):
                color = grid[i][j]
                neighbors = []
                for di in range(-window, window+1):
                    for dj in range(-window, window+1):
                        ni, nj = i+di, j+dj
                        if (0 <= ni < h) and (0 <= nj < w) and (di or dj):
                            neighbors.append(grid[ni][nj])
                for n in neighbors:
                    co_occur[(color, n)] += 1
    
    kwic_el = ET.Element("kwic")
    kwic_el.set("total_cells", str(total_cells))
    kwic_el.set("grid_count", str(len(grids)))
    kwic_el.set("window_size", str(window))
    
    # Sort by frequency for easier analysis
    sorted_pairs = sorted(co_occur.items(), key=lambda x: x[1], reverse=True)
    for (c1, c2), cnt in sorted_pairs:
        ET.SubElement(kwic_el, "pair", 
                     color1=str(c1), 
                     color2=str(c2), 
                     count=str(cnt),
                     frequency=f"{cnt/total_cells:.4f}")
    
    return kwic_el

def main():
    # --- Load JSON ---
    with open(TRAIN_CHALLENGES_JSON, "r") as f:
        challenges = json.load(f)
    with open(TRAIN_SOLUTIONS_JSON, "r") as f:
        solutions = json.load(f)

    # --- Compose root element ---
    root = ET.Element("arc_agi_tasks")

    for tid, task_data in challenges.items():
        train_pairs = task_data["train"]
        test_pairs = task_data.get("test", [])

        # Metadata collection
        colors = set()
        input_sizes = []
        output_sizes = []
        transformations = []
        all_histograms = []
        transformation_hints = []
        for pair in train_pairs:
            in_stats = parse_grid(pair["input"])
            out_stats = parse_grid(pair["output"])
            colors.update(in_stats["colors"])
            colors.update(out_stats["colors"])
            input_sizes.append(in_stats)
            output_sizes.append(out_stats)
            all_histograms.append(in_stats["histogram"])
            transformations.append(
                f"{in_stats['height']}×{in_stats['width']} → {out_stats['height']}×{out_stats['width']}"
            )
            transformation_hints.append(hypothesize_transformation(in_stats, out_stats))

        meta = make_metadata_element(
            colors=colors,
            grid_sizes={"inputs": input_sizes, "outputs": output_sizes},
            trans_list=transformations,
            all_histograms=all_histograms,
            transformation_hints=transformation_hints
        )

        task_el = ET.SubElement(root, "arc_agi_task", id=str(tid))
        task_el.append(meta)

        # --- Training examples ---
        train_el = ET.SubElement(task_el, "training_examples", count=str(len(train_pairs)))
        for idx, pair in enumerate(train_pairs):
            ex_el = ET.SubElement(train_el, "example", index=str(idx))
            
            # Input with individual KWIC analysis
            in_grid = parse_grid(pair["input"])
            in_el = ET.SubElement(ex_el, "input", height=str(in_grid["height"]), width=str(in_grid["width"]))
            for ridx, row in enumerate(in_grid["rows"]):
                ET.SubElement(in_el, "row", index=str(ridx)).text = row
            
            # Add KWIC analysis for this specific input
            input_kwic = kwic_for_colors([pair["input"]])
            input_kwic.set("type", "training_input")
            input_kwic.set("example_index", str(idx))
            in_el.append(input_kwic)
            
            # Output
            out_grid = parse_grid(pair["output"])
            out_el = ET.SubElement(ex_el, "output", height=str(out_grid["height"]), width=str(out_grid["width"]))
            for ridx, row in enumerate(out_grid["rows"]):
                ET.SubElement(out_el, "row", index=str(ridx)).text = row

        # --- Test examples (if present) ---
        if test_pairs:
            test_el = ET.SubElement(task_el, "test_examples", count=str(len(test_pairs)))
            for idx, pair in enumerate(test_pairs):
                ex_el = ET.SubElement(test_el, "example", index=str(idx))
                
                # Input with individual KWIC analysis
                in_grid = parse_grid(pair["input"])
                in_el = ET.SubElement(ex_el, "input", height=str(in_grid["height"]), width=str(in_grid["width"]))
                for ridx, row in enumerate(in_grid["rows"]):
                    ET.SubElement(in_el, "row", index=str(ridx)).text = row
                
                # Add KWIC analysis for this specific input
                input_kwic = kwic_for_colors([pair["input"]])
                input_kwic.set("type", "test_input")
                input_kwic.set("example_index", str(idx))
                in_el.append(input_kwic)
                # NOTE: output is not available for test in real challenge

        # --- Add KWIC color context for richer pattern analysis ---
        # Use only input grids for KWIC analysis (consistent across training and competition)
        input_grids = [p["input"] for p in train_pairs]
        kwic_el = kwic_for_colors(input_grids)
        
        if train_pairs and "output" in train_pairs[0]:
            # Training mode: we have outputs available for learning, but KWIC uses inputs only
            kwic_el.set("type", "training")  # Mark as training data
        else:
            # Competition mode: only inputs available
            kwic_el.set("type", "competition")  # Mark as competition data
        
        task_el.find("metadata").append(kwic_el)

    # --- Save output ---
    os.makedirs(os.path.dirname(OUTPUT_XML), exist_ok=True)
    if os.path.exists(OUTPUT_XML):
        try:
            os.remove(OUTPUT_XML)
            print(f"Deleted existing file: {OUTPUT_XML}")
        except Exception as e:
            print(f"Warning: Could not delete existing file {OUTPUT_XML}: {e}")

    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")
    except Exception:
        pass
    tree.write(OUTPUT_XML, encoding="utf-8", xml_declaration=True)
    print(f"XML written: {OUTPUT_XML}")

def process_test_data():
    """
    Process test challenges (competition mode) - no solutions available.
    This demonstrates the type="competition" KWIC analysis.
    """
    print("Processing test data (competition mode)...")
    
    # --- Load test challenges (no solutions) ---
    with open(TEST_CHALLENGES_JSON, "r") as f:
        test_challenges = json.load(f)

    # --- Compose root element ---
    root = ET.Element("arc_agi_tasks", mode="competition")

    for tid, task_data in test_challenges.items():
        train_pairs = task_data["train"]
        test_pairs = task_data["test"]
        
        # --- Build task element ---
        task_el = ET.SubElement(root, "arc_agi_task", id=tid)
        
        # --- Collect metadata from training examples (these have solutions) ---
        all_colors = set()
        all_histograms = []
        grid_sizes = {"inputs": [], "outputs": []}
        trans_list = []
        transformation_hints = []
        
        for pair in train_pairs:
            in_grid = parse_grid(pair["input"])
            out_grid = parse_grid(pair["output"])
            all_colors.update(in_grid["colors"])
            all_colors.update(out_grid["colors"])
            all_histograms.extend([in_grid["histogram"], out_grid["histogram"]])
            grid_sizes["inputs"].append({"height": in_grid["height"], "width": in_grid["width"], "area": in_grid["area"]})
            grid_sizes["outputs"].append({"height": out_grid["height"], "width": out_grid["width"], "area": out_grid["area"]})
            trans = hypothesize_transformation(in_grid, out_grid)
            if trans:
                trans_list.append(trans)
                transformation_hints.append(trans)
        
        # --- Add metadata ---
        task_el.append(make_metadata_element(all_colors, grid_sizes, trans_list, all_histograms, transformation_hints))
        
        # --- Add training examples ---
        train_ex_el = ET.SubElement(task_el, "training_examples")
        for idx, pair in enumerate(train_pairs):
            ex_el = ET.SubElement(train_ex_el, "example", index=str(idx))
            
            # Input with individual KWIC analysis
            in_grid = parse_grid(pair["input"])
            in_el = ET.SubElement(ex_el, "input", height=str(in_grid["height"]), width=str(in_grid["width"]))
            for ridx, row in enumerate(in_grid["rows"]):
                ET.SubElement(in_el, "row", index=str(ridx)).text = row
            
            # Add KWIC analysis for this specific training input
            input_kwic = kwic_for_colors([pair["input"]])
            input_kwic.set("type", "training_input")
            input_kwic.set("example_index", str(idx))
            in_el.append(input_kwic)
            
            # Output (available for training examples)
            out_grid = parse_grid(pair["output"])
            out_el = ET.SubElement(ex_el, "output", height=str(out_grid["height"]), width=str(out_grid["width"]))
            for ridx, row in enumerate(out_grid["rows"]):
                ET.SubElement(out_el, "row", index=str(ridx)).text = row
        
        # --- Add test examples (competition mode - no solutions) ---
        test_ex_el = ET.SubElement(task_el, "test_examples")
        for idx, pair in enumerate(test_pairs):
            ex_el = ET.SubElement(test_ex_el, "example", index=str(idx))
            
            # Only input available in competition mode, with individual KWIC analysis
            in_grid = parse_grid(pair["input"])
            in_el = ET.SubElement(ex_el, "input", height=str(in_grid["height"]), width=str(in_grid["width"]))
            for ridx, row in enumerate(in_grid["rows"]):
                ET.SubElement(in_el, "row", index=str(ridx)).text = row
                
            # Add KWIC analysis for this specific test input
            input_kwic = kwic_for_colors([pair["input"]])
            input_kwic.set("type", "test_input")
            input_kwic.set("example_index", str(idx))
            in_el.append(input_kwic)
            # NOTE: No output available for test examples in competition mode
        
        # --- Add KWIC analysis ---
        # For test examples: only input grids available (competition mode)
        test_input_grids = [p["input"] for p in test_pairs]
        kwic_el = kwic_for_colors(test_input_grids)
        kwic_el.set("type", "competition")  # Mark as competition data
        task_el.find("metadata").append(kwic_el)

    # --- Save test XML ---
    os.makedirs(os.path.dirname(TEST_OUTPUT_XML), exist_ok=True)
    if os.path.exists(TEST_OUTPUT_XML):
        try:
            os.remove(TEST_OUTPUT_XML)
            print(f"Deleted existing file: {TEST_OUTPUT_XML}")
        except Exception as e:
            print(f"Warning: Could not delete existing file {TEST_OUTPUT_XML}: {e}")

    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")
    except Exception:
        pass
    tree.write(TEST_OUTPUT_XML, encoding="utf-8", xml_declaration=True)
    print(f"Test XML written: {TEST_OUTPUT_XML}")

if __name__ == "__main__":
    # Process training data (has both inputs and outputs)
    main()
    
    # Process test data (competition mode - inputs only)
    process_test_data()
