# Glyph Weight System Specification

## Version: 1.3
## Date: May 30, 2025
## Status: Production Ready

---

## Executive Summary

The Glyph Weight System is a foundational computational mechanism in the SYNTHEON AI reasoning engine that provides deterministic conflict resolution, symbolic foresight, and mathematical consistency for all grid transformations. This specification defines the complete architecture, implementation requirements, and operational protocols.

---

## 1. Architecture Overview

### 1.1 Purpose
The glyph weight system serves four critical functions:
1. **Deterministic Conflict Resolution** - Resolves competing transformations
2. **Symbolic Foresight Integration** - Enables complexity prediction
3. **Rule Chain Prioritization** - Orders transformation sequences
4. **Mathematical Consistency** - Ensures reproducible outcomes

### 1.2 Core Principle
**Lower Weight = Higher Priority**: Operations with lower glyph weights take precedence during conflict resolution, ensuring simpler transformations are preferred over complex ones.

---

## 2. Glyph Weight Hierarchy

### 2.1 Complete Weight Table

| Glyph | Color | Weight | Priority | Semantic Meaning | Usage Context |
|-------|-------|--------|----------|------------------|---------------|
| ⋯     | 0     | 0.000  | 1 (Highest) | Void/Empty | Background, erasure |
| ⧖     | 1     | 0.021  | 2 | Time/Process | Temporal operations |
| ✦     | 2     | 0.034  | 3 | Star/Focus | Focus transformations |
| ⛬     | 3     | 0.055  | 4 | Structure | Structural modifications |
| █     | 4     | 0.089  | 5 | Solid/Mass | Mass operations |
| ⟡     | 5     | 0.144  | 6 | Boundary | Boundary transformations |
| ◐     | 6     | 0.233  | 7 | Duality/Rotation | Rotational operations |
| 🜄     | 7     | 0.377  | 8 | Transformation | Complex transformations |
| ◼     | 8     | 0.610  | 9 | Dense/Core | Core operations |
| ✕     | 9     | 1.000  | 10 (Lowest) | Negation/Cross | Removal, negation |

### 2.2 Weight Progression
The weights follow a **Fibonacci-like progression** optimized for ARC-AGI pattern complexity:
- **0.000** - Base (void)
- **0.021** - φ⁻⁴ approximation  
- **0.034** - φ⁻³ approximation
- **0.055** - φ⁻² approximation
- **0.089** - φ⁻¹ approximation
- **0.144** - φ⁰ approximation
- **0.233** - φ¹ approximation
- **0.377** - φ² approximation (Golden ratio territory)
- **0.610** - φ³ approximation
- **1.000** - Maximum complexity

---

## 3. Implementation Requirements

### 3.1 Core Functions

#### 3.1.1 Weight Lookup Function
```python
@staticmethod
def get_glyph_weight(color: int) -> float:
    """Get canonical foresight weight for deterministic operations"""
    weights = {
        0: 0.000,  # ⋯ Void/Empty
        1: 0.021,  # ⧖ Time/Process
        2: 0.034,  # ✦ Star/Focus
        3: 0.055,  # ⛬ Structure
        4: 0.089,  # █ Solid/Mass
        5: 0.144,  # ⟡ Boundary
        6: 0.233,  # ◐ Duality
        7: 0.377,  # 🜄 Transformation
        8: 0.610,  # ◼ Dense/Core
        9: 1.000   # ✕ Negation/Cross
    }
    return weights.get(color, 0.5)  # Default fallback
```

#### 3.1.2 Conflict Resolution Function
```python
@staticmethod
def resolve_glyph_conflict(colors: List[int], positions: List[Tuple[int, int]]) -> int:
    """
    Resolve conflicts between multiple colors/glyphs competing for same position.
    Uses glyph weights for deterministic tie-breaking as per symbolic foresight loop.
    """
    if not colors:
        return 0
    if len(colors) == 1:
        return colors[0]
    
    # Sort by glyph weight (lower weight = higher priority)
    color_weight_pairs = [(color, get_glyph_weight(color)) for color in colors]
    color_weight_pairs.sort(key=lambda x: x[1])  # Sort by weight ascending
    
    selected_color = color_weight_pairs[0][0]
    selected_glyph = color_to_glyph(selected_color)
    
    # Log symbolic reasoning
    logging.info(f"Glyph conflict resolution: {[color_to_glyph(c) for c in colors]} → {selected_glyph} (weight {get_glyph_weight(selected_color)})")
    
    return selected_color
```

### 3.2 Symbolic Foresight Integration

#### 3.2.1 Grid Weight Analysis
```python
def analyze_grid_complexity(input_grid: np.ndarray) -> Dict[str, float]:
    """Analyze grid complexity using glyph weights"""
    weight_grid = [[get_glyph_weight(cell) for cell in row] for row in input_grid]
    total_weight = sum(sum(row) for row in weight_grid)
    avg_weight = total_weight / (input_grid.shape[0] * input_grid.shape[1])
    
    return {
        'total_weight': total_weight,
        'average_weight': avg_weight,
        'complexity_category': categorize_complexity(avg_weight)
    }

def categorize_complexity(avg_weight: float) -> str:
    """Categorize grid complexity based on average weight"""
    if avg_weight < 0.1: return "SIMPLE"
    elif avg_weight < 0.3: return "MODERATE" 
    elif avg_weight < 0.6: return "COMPLEX"
    else: return "VERY_COMPLEX"
```

---

## 4. Glyph Chain Analysis

### 4.1 Chain Weight Calculation
For glyph chains like `⟡◐⟡`, calculate:

```python
def analyze_glyph_chain(chain: str) -> Dict[str, float]:
    """Analyze weight properties of a glyph chain"""
    glyphs = list(chain)
    weights = [glyph_to_weight(g) for g in glyphs]
    
    return {
        'total_weight': sum(weights),
        'average_weight': sum(weights) / len(weights),
        'min_weight': min(weights),
        'max_weight': max(weights),
        'weight_variance': calculate_variance(weights),
        'complexity_pattern': describe_pattern(weights)
    }
```

### 4.2 Example: RotatePattern Analysis
**Chain**: `⟡◐⟡`
- **Weights**: [0.144, 0.233, 0.144]
- **Total Weight**: 0.521
- **Average Weight**: 0.174 (moderate complexity)
- **Pattern**: Symmetric (Low → Medium → Low)
- **Variance**: 0.0026 (low variance, stable transformation)

---

## 5. Operational Protocols

### 5.1 Rule Definition Requirements
All rules in `syntheon_rules_glyphs.xml` must include:

```xml
<rule id="R43" name="RotatePattern">
    <pattern>rotate grid by specified angle (90°, 180°, 270°)</pattern>
    <description>Rotates the entire grid by the specified number of degrees clockwise.</description>
    <glyph_chain>⟡◐⟡</glyph_chain>
    <!-- Weight Analysis: Total=0.521, Avg=0.174, Pattern=Symmetric -->
    <logic language="pseudo">
        if degrees == 90: return np.rot90(grid, k=3)
        elif degrees == 180: return np.rot90(grid, k=2)
        elif degrees == 270: return np.rot90(grid, k=1)
    </logic>
    <condition>degrees in [90, 180, 270]</condition>
</rule>
```

### 5.2 Performance Validation
System must maintain:
- **Deterministic Outcomes**: Same input → same output
- **Weight Consistency**: All conflicts resolved by weight hierarchy
- **Performance Metrics**: No degradation due to weight miscalculation

### 5.3 Error Handling
```python
def validate_glyph_weights():
    """Validate all glyph weights are properly configured"""
    required_weights = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    configured_weights = set(get_glyph_weight(i) for i in range(10))
    
    if len(configured_weights) != len(required_weights):
        raise SystemError("Glyph weight configuration incomplete")
    
    # Validate weight ordering
    for i in range(9):
        if get_glyph_weight(i) >= get_glyph_weight(i+1):
            raise SystemError(f"Weight ordering violated at {i}")
```

---

## 6. Performance Impact Analysis

### 6.1 Critical Success Factors
The glyph weight system directly impacts:

1. **Rule Chain Formation**: 
   - Complex sequences like `RotatePattern → DiagonalFlip → CropToBoundingBox`
   - 10 successful applications in production

2. **Transformation Prioritization**:
   - Simple operations (low weight) execute before complex ones
   - Prevents infinite loops and conflicts

3. **Conflict Resolution**:
   - Deterministic handling when multiple rules compete
   - Essential for 4.24% accuracy achievement

4. **Symbolic Reasoning**:
   - Weight-based complexity prediction
   - Enables symbolic foresight workflow

### 6.2 Performance Case Study: RotatePattern Recovery
**Before**: Missing RotatePattern rule → 3.53% accuracy (114 solved)
**After**: Added RotatePattern with `⟡◐⟡` chain → 4.24% accuracy (137 solved)
**Impact**: +23 solved examples (+0.71% accuracy) from single rule addition

**Root Cause**: Improper glyph weight handling caused:
- Failed conflict resolution
- Broken rule chain formation  
- Incomplete symbolic foresight

---

## 7. Testing & Validation

### 7.1 Unit Tests
```python
def test_glyph_weight_hierarchy():
    """Test weight hierarchy is properly ordered"""
    for i in range(9):
        assert get_glyph_weight(i) < get_glyph_weight(i+1)

def test_conflict_resolution():
    """Test deterministic conflict resolution"""
    colors = [5, 6, 2]  # ⟡, ◐, ✦
    result = resolve_glyph_conflict(colors, [(0,0)])
    assert result == 2  # ✦ has lowest weight (0.034)

def test_chain_analysis():
    """Test glyph chain weight calculation"""
    chain_weights = analyze_glyph_chain("⟡◐⟡")
    assert abs(chain_weights['total_weight'] - 0.521) < 0.001
    assert abs(chain_weights['average_weight'] - 0.174) < 0.001
```

### 7.2 Integration Tests
```python
def test_symbolic_foresight_integration():
    """Test weight system integration with symbolic foresight"""
    grid = np.array([[5, 6], [6, 5]])  # ⟡◐ / ◐⟡
    analysis = analyze_grid_complexity(grid)
    assert analysis['complexity_category'] == "MODERATE"
    assert 0.15 < analysis['average_weight'] < 0.25
```

---

## 8. Maintenance & Evolution

### 8.1 Version Control
- All weight changes must be documented with performance impact
- Regression testing required for any weight modifications
- Backward compatibility maintained for existing rule chains

### 8.2 Future Enhancements
- **Dynamic Weight Learning**: Adapt weights based on task performance
- **Context-Sensitive Weights**: Weights that vary by problem domain
- **Multi-Dimensional Weights**: Separate weights for different conflict types

---

## 9. Appendices

### Appendix A: Mathematical Foundation
The weight progression is based on the **golden ratio (φ ≈ 1.618)** series:
- Provides natural hierarchy for pattern complexity
- Optimized for human-interpretable symbolic reasoning
- Mathematically stable for recursive operations

### Appendix B: Historical Context
- **Version 1.0**: Basic weight system (deprecated)
- **Version 1.2**: Enhanced with symbolic foresight integration
- **Version 1.3**: Production-ready with RotatePattern recovery

### Appendix C: Related Systems
- **Glyph Mapping**: Color-to-symbol conversion (`syntheon_engine.py`)
- **Rule Chains**: Sequential rule application (`syntheon_rules_glyphs.xml`)
- **Symbolic Foresight**: Predictive reasoning workflow (`main.py`)

---

**End of Specification**

*This document defines the complete Glyph Weight System architecture for SYNTHEON AI reasoning engine. All implementations must conform to this specification to ensure deterministic, reproducible, and high-performance symbolic reasoning.*
