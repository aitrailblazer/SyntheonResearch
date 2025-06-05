"""
Enhanced Preprocessing System for ARC Challenge
Spatial Pattern Analysis Module (SPAM) - Phase 1 Implementation

This module implements advanced pattern detection beyond KWIC to improve
rule determination from input analysis alone.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import Counter
import math

@dataclass
class TilingAnalysis:
    """Results of tiling pattern analysis"""
    is_tiling: bool
    vertical_tiles: int
    horizontal_tiles: int
    total_tiles: int
    tile_transformations: List[Dict[str, Any]]
    tiling_confidence: float

@dataclass
class ScalingAnalysis:
    """Results of scaling pattern analysis"""
    height_ratio: float
    width_ratio: float
    is_uniform_scaling: bool
    is_integer_scaling: bool
    scale_factor: Optional[int]
    scaling_type: str  # 'exact', 'approximate', 'complex'

@dataclass
class GeometricTransformation:
    """Detected geometric transformation"""
    transformation_type: str
    confidence: float
    parameters: Dict[str, Any]

class SpatialPatternAnalyzer:
    """Advanced spatial pattern analysis for ARC tasks"""
    
    def __init__(self):
        self.transformation_cache = {}
    
    def analyze_scaling_patterns(self, input_grid: List[List[int]], 
                               output_grid: List[List[int]]) -> ScalingAnalysis:
        """Detect size transformations and scaling ratios"""
        input_h, input_w = len(input_grid), len(input_grid[0])
        output_h, output_w = len(output_grid), len(output_grid[0])
        
        height_ratio = output_h / input_h
        width_ratio = output_w / input_w
        is_uniform = abs(height_ratio - width_ratio) < 0.001
        is_integer = (output_h % input_h == 0) and (output_w % input_w == 0)
        
        scale_factor = None
        scaling_type = "complex"
        
        if is_integer and is_uniform:
            scale_factor = int(height_ratio)
            scaling_type = "exact"
        elif is_integer:
            scaling_type = "approximate"
        
        return ScalingAnalysis(
            height_ratio=height_ratio,
            width_ratio=width_ratio,
            is_uniform_scaling=is_uniform,
            is_integer_scaling=is_integer,
            scale_factor=scale_factor,
            scaling_type=scaling_type
        )
    
    def detect_tiling_patterns(self, input_grid: List[List[int]], 
                             output_grid: List[List[int]]) -> TilingAnalysis:
        """Identify if output is composed of tiled input patterns"""
        input_h, input_w = len(input_grid), len(input_grid[0])
        output_h, output_w = len(output_grid), len(output_grid[0])
        
        if output_h % input_h != 0 or output_w % input_w != 0:
            return TilingAnalysis(
                is_tiling=False, vertical_tiles=0, horizontal_tiles=0, 
                total_tiles=0, tile_transformations=[], tiling_confidence=0.0
            )
        
        tiles_v = output_h // input_h
        tiles_h = output_w // input_w
        
        tile_transformations = []
        matching_tiles = 0
        
        # Analyze each tile for transformations
        for i in range(tiles_v):
            for j in range(tiles_h):
                tile = self._extract_tile(output_grid, i, j, input_h, input_w)
                transformation = self._identify_tile_transformation(input_grid, tile)
                tile_transformations.append({
                    'position': (i, j),
                    'transformation': transformation.transformation_type,
                    'confidence': transformation.confidence,
                    'parameters': transformation.parameters
                })
                
                if transformation.confidence > 0.8:
                    matching_tiles += 1
        
        total_tiles = tiles_v * tiles_h
        tiling_confidence = matching_tiles / total_tiles if total_tiles > 0 else 0.0
        
        return TilingAnalysis(
            is_tiling=True,
            vertical_tiles=tiles_v,
            horizontal_tiles=tiles_h,
            total_tiles=total_tiles,
            tile_transformations=tile_transformations,
            tiling_confidence=tiling_confidence
        )
    
    def _extract_tile(self, grid: List[List[int]], tile_i: int, tile_j: int, 
                     tile_h: int, tile_w: int) -> List[List[int]]:
        """Extract a tile from the grid at specified position"""
        start_row = tile_i * tile_h
        start_col = tile_j * tile_w
        
        tile = []
        for row in range(start_row, start_row + tile_h):
            tile_row = []
            for col in range(start_col, start_col + tile_w):
                tile_row.append(grid[row][col])
            tile.append(tile_row)
        
        return tile
    
    def _identify_tile_transformation(self, original: List[List[int]], 
                                    tile: List[List[int]]) -> GeometricTransformation:
        """Identify what transformation was applied to create this tile"""
        if not original or not tile:
            return GeometricTransformation("none", 0.0, {})
        
        # Check for exact match
        if self._grids_equal(original, tile):
            return GeometricTransformation("identity", 1.0, {})
        
        # Check for horizontal flip
        h_flipped = self._horizontal_flip(original)
        if self._grids_equal(h_flipped, tile):
            return GeometricTransformation("horizontal_flip", 1.0, {})
        
        # Check for vertical flip  
        v_flipped = self._vertical_flip(original)
        if self._grids_equal(v_flipped, tile):
            return GeometricTransformation("vertical_flip", 1.0, {})
        
        # Check for 180-degree rotation
        rotated_180 = self._rotate_180(original)
        if self._grids_equal(rotated_180, tile):
            return GeometricTransformation("rotation_180", 1.0, {})
        
        # Check for 90-degree rotations (if square)
        if len(original) == len(original[0]):
            rotated_90 = self._rotate_90(original)
            if self._grids_equal(rotated_90, tile):
                return GeometricTransformation("rotation_90", 1.0, {})
            
            rotated_270 = self._rotate_270(original)
            if self._grids_equal(rotated_270, tile):
                return GeometricTransformation("rotation_270", 1.0, {})
        
        # Check for partial matches (could be complex transformation)
        similarity = self._calculate_similarity(original, tile)
        if similarity > 0.5:
            return GeometricTransformation("complex", similarity, {"similarity": similarity})
        
        return GeometricTransformation("unknown", 0.0, {})
    
    def _grids_equal(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """Check if two grids are identical"""
        if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
            return False
        
        for i in range(len(grid1)):
            for j in range(len(grid1[0])):
                if grid1[i][j] != grid2[i][j]:
                    return False
        return True
    
    def _horizontal_flip(self, grid: List[List[int]]) -> List[List[int]]:
        """Flip grid horizontally"""
        return [row[::-1] for row in grid]
    
    def _vertical_flip(self, grid: List[List[int]]) -> List[List[int]]:
        """Flip grid vertically"""
        return grid[::-1]
    
    def _rotate_90(self, grid: List[List[int]]) -> List[List[int]]:
        """Rotate grid 90 degrees clockwise"""
        rows, cols = len(grid), len(grid[0])
        rotated = [[0] * rows for _ in range(cols)]
        
        for i in range(rows):
            for j in range(cols):
                rotated[j][rows - 1 - i] = grid[i][j]
        
        return rotated
    
    def _rotate_180(self, grid: List[List[int]]) -> List[List[int]]:
        """Rotate grid 180 degrees"""
        return [row[::-1] for row in grid[::-1]]
    
    def _rotate_270(self, grid: List[List[int]]) -> List[List[int]]:
        """Rotate grid 270 degrees clockwise"""
        rows, cols = len(grid), len(grid[0])
        rotated = [[0] * rows for _ in range(cols)]
        
        for i in range(rows):
            for j in range(cols):
                rotated[cols - 1 - j][i] = grid[i][j]
        
        return rotated
    
    def _calculate_similarity(self, grid1: List[List[int]], 
                            grid2: List[List[int]]) -> float:
        """Calculate similarity between two grids"""
        if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
            return 0.0
        
        total_cells = len(grid1) * len(grid1[0])
        matching_cells = 0
        
        for i in range(len(grid1)):
            for j in range(len(grid1[0])):
                if grid1[i][j] == grid2[i][j]:
                    matching_cells += 1
        
        return matching_cells / total_cells


class EnhancedInputFeatureExtractor:
    """Extract features from input that could predict transformation type"""
    
    def extract_input_only_features(self, input_grid: List[List[int]]) -> Dict[str, Any]:
        """Extract comprehensive features from input grid alone"""
        flat_grid = [cell for row in input_grid for cell in row]
        color_counts = Counter(flat_grid)
        
        features = {
            # Size and shape features
            'height': len(input_grid),
            'width': len(input_grid[0]),
            'total_cells': len(flat_grid),
            'aspect_ratio': len(input_grid[0]) / len(input_grid),
            'is_square': len(input_grid) == len(input_grid[0]),
            'is_small': len(flat_grid) <= 9,  # 3x3 or smaller
            
            # Color distribution features
            'unique_colors': len(color_counts),
            'dominant_color': max(color_counts, key=color_counts.get),
            'color_entropy': self._calculate_entropy(list(color_counts.values())),
            'color_balance': self._calculate_color_balance(color_counts),
            'max_color_frequency': max(color_counts.values()) / len(flat_grid),
            
            # Structural features
            'corner_diversity': self._analyze_corners(input_grid),
            'edge_complexity': self._analyze_edges(input_grid),
            'has_repeated_rows': self._has_repeated_rows(input_grid),
            'has_repeated_cols': self._has_repeated_columns(input_grid),
            'diagonal_symmetry': self._check_diagonal_symmetry(input_grid),
            'horizontal_symmetry': self._check_horizontal_symmetry(input_grid),
            'vertical_symmetry': self._check_vertical_symmetry(input_grid),
            
            # Pattern complexity indicators
            'local_variation': self._calculate_local_variation(input_grid),
            'predictability_score': self._assess_predictability(input_grid),
            'tiling_potential': self._assess_tiling_potential(input_grid)
        }
        
        return features
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """Calculate Shannon entropy of value distribution"""
        if not values:
            return 0.0
        
        total = sum(values)
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in values:
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def _calculate_color_balance(self, color_counts: Counter) -> float:
        """Calculate how balanced the color distribution is (0=uniform, 1=one dominant)"""
        if len(color_counts) <= 1:
            return 1.0
        
        values = list(color_counts.values())
        total = sum(values)
        max_freq = max(values)
        
        return max_freq / total
    
    def _analyze_corners(self, grid: List[List[int]]) -> int:
        """Count unique colors in corners"""
        if len(grid) < 2 or len(grid[0]) < 2:
            return len(set([cell for row in grid for cell in row]))
        
        corners = [
            grid[0][0], grid[0][-1],  # top corners
            grid[-1][0], grid[-1][-1]  # bottom corners
        ]
        return len(set(corners))
    
    def _analyze_edges(self, grid: List[List[int]]) -> float:
        """Analyze complexity of edge patterns"""
        if len(grid) < 2 or len(grid[0]) < 2:
            return 0.0
        
        edges = []
        # Top and bottom edges
        edges.extend(grid[0])
        edges.extend(grid[-1])
        # Left and right edges (excluding corners already counted)
        for i in range(1, len(grid) - 1):
            edges.extend([grid[i][0], grid[i][-1]])
        
        return self._calculate_entropy(list(Counter(edges).values()))
    
    def _has_repeated_rows(self, grid: List[List[int]]) -> bool:
        """Check if any rows are repeated"""
        row_strings = [str(row) for row in grid]
        return len(row_strings) != len(set(row_strings))
    
    def _has_repeated_columns(self, grid: List[List[int]]) -> bool:
        """Check if any columns are repeated"""
        cols = []
        for j in range(len(grid[0])):
            col = [grid[i][j] for i in range(len(grid))]
            cols.append(str(col))
        return len(cols) != len(set(cols))
    
    def _check_diagonal_symmetry(self, grid: List[List[int]]) -> bool:
        """Check for diagonal symmetry (main diagonal)"""
        if len(grid) != len(grid[0]):
            return False
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] != grid[j][i]:
                    return False
        return True
    
    def _check_horizontal_symmetry(self, grid: List[List[int]]) -> bool:
        """Check for horizontal (left-right) symmetry"""
        for row in grid:
            if row != row[::-1]:
                return False
        return True
    
    def _check_vertical_symmetry(self, grid: List[List[int]]) -> bool:
        """Check for vertical (top-bottom) symmetry"""
        return grid == grid[::-1]
    
    def _calculate_local_variation(self, grid: List[List[int]]) -> float:
        """Calculate how much adjacent cells differ"""
        if len(grid) < 2 or len(grid[0]) < 2:
            return 0.0
        
        differences = 0
        total_comparisons = 0
        
        # Check horizontal adjacencies
        for i in range(len(grid)):
            for j in range(len(grid[0]) - 1):
                if grid[i][j] != grid[i][j+1]:
                    differences += 1
                total_comparisons += 1
        
        # Check vertical adjacencies
        for i in range(len(grid) - 1):
            for j in range(len(grid[0])):
                if grid[i][j] != grid[i+1][j]:
                    differences += 1
                total_comparisons += 1
        
        return differences / total_comparisons if total_comparisons > 0 else 0.0
    
    def _assess_predictability(self, grid: List[List[int]]) -> float:
        """Assess how predictable the pattern is"""
        # Combine various factors that indicate predictability
        factors = []
        
        # Symmetry increases predictability
        if self._check_horizontal_symmetry(grid):
            factors.append(0.3)
        if self._check_vertical_symmetry(grid):
            factors.append(0.3)
        if self._check_diagonal_symmetry(grid):
            factors.append(0.3)
        
        # Repeated elements increase predictability
        if self._has_repeated_rows(grid):
            factors.append(0.2)
        if self._has_repeated_columns(grid):
            factors.append(0.2)
        
        # Low local variation increases predictability
        local_var = self._calculate_local_variation(grid)
        factors.append(1.0 - local_var)
        
        return sum(factors) / len(factors) if factors else 0.0
    
    def _assess_tiling_potential(self, grid: List[List[int]]) -> float:
        """Assess how likely this input is to be used for tiling"""
        potential = 0.0
        
        # Small grids are more likely to be tiled
        size = len(grid) * len(grid[0])
        if size <= 4:
            potential += 0.4
        elif size <= 9:
            potential += 0.2
        
        # High color diversity suggests tiling potential
        flat_grid = [cell for row in grid for cell in row]
        unique_colors = len(set(flat_grid))
        if unique_colors >= len(flat_grid) // 2:
            potential += 0.3
        
        # Asymmetric patterns are often tiled with transformations
        symmetries = [
            self._check_horizontal_symmetry(grid),
            self._check_vertical_symmetry(grid),
            self._check_diagonal_symmetry(grid)
        ]
        if not any(symmetries):
            potential += 0.3
        
        return min(potential, 1.0)


class TransformationTypePredictor:
    """Predict likely transformation types from input features"""
    
    def __init__(self):
        # Transformation signatures based on successful ARC patterns
        self.transformation_signatures = {
            'tiling_with_mirroring': {
                'size_requirements': {'max_area': 9, 'is_small': True},
                'color_requirements': {'min_unique': 3, 'high_entropy': True},
                'structural_requirements': {'asymmetric': True, 'high_variation': True},
                'priority_boost': 0.8  # High priority for task 00576224 type
            },
            'simple_tiling': {
                'size_requirements': {'max_area': 16, 'is_small': True},
                'color_requirements': {'min_unique': 2},
                'structural_requirements': {'predictable': True},
                'priority_boost': 0.6
            },
            'scaling_transformation': {
                'size_requirements': {'max_area': 25},
                'color_requirements': {'balanced': True},
                'structural_requirements': {'symmetric': True},
                'priority_boost': 0.4
            },
            'geometric_transformation': {
                'size_requirements': {'is_square': True},
                'color_requirements': {'diverse': True},
                'structural_requirements': {'complex': True},
                'priority_boost': 0.5
            }
        }
    
    def predict_transformation_type(self, input_features: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Predict transformation types with confidence scores"""
        predictions = {}
        
        for transform_type, signature in self.transformation_signatures.items():
            score = self._calculate_match_score(input_features, signature)
            predictions[transform_type] = score
        
        return sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    def _calculate_match_score(self, features: Dict[str, Any], 
                             signature: Dict[str, Any]) -> float:
        """Calculate how well input features match transformation signature"""
        score = 0.0
        total_weight = 0.0
        
        # Check size requirements
        if 'size_requirements' in signature:
            size_score, size_weight = self._evaluate_size_requirements(
                features, signature['size_requirements']
            )
            score += size_score * size_weight
            total_weight += size_weight
        
        # Check color requirements  
        if 'color_requirements' in signature:
            color_score, color_weight = self._evaluate_color_requirements(
                features, signature['color_requirements']
            )
            score += color_score * color_weight
            total_weight += color_weight
        
        # Check structural requirements
        if 'structural_requirements' in signature:
            struct_score, struct_weight = self._evaluate_structural_requirements(
                features, signature['structural_requirements']
            )
            score += struct_score * struct_weight
            total_weight += struct_weight
        
        # Apply priority boost
        base_score = score / total_weight if total_weight > 0 else 0.0
        priority_boost = signature.get('priority_boost', 0.0)
        
        return base_score * (1.0 + priority_boost)
    
    def _evaluate_size_requirements(self, features: Dict[str, Any], 
                                  requirements: Dict[str, Any]) -> Tuple[float, float]:
        """Evaluate size-based requirements"""
        score = 0.0
        count = 0
        
        if 'max_area' in requirements:
            if features['total_cells'] <= requirements['max_area']:
                score += 1.0
            count += 1
        
        if 'is_small' in requirements:
            if features['is_small'] == requirements['is_small']:
                score += 1.0
            count += 1
        
        if 'is_square' in requirements:
            if features['is_square'] == requirements['is_square']:
                score += 1.0
            count += 1
        
        return (score / count if count > 0 else 0.0, 2.0)  # High weight for size
    
    def _evaluate_color_requirements(self, features: Dict[str, Any], 
                                   requirements: Dict[str, Any]) -> Tuple[float, float]:
        """Evaluate color-based requirements"""
        score = 0.0
        count = 0
        
        if 'min_unique' in requirements:
            if features['unique_colors'] >= requirements['min_unique']:
                score += 1.0
            count += 1
        
        if 'high_entropy' in requirements:
            # High entropy threshold for colors
            if features['color_entropy'] > 1.5:
                score += 1.0
            count += 1
        
        if 'balanced' in requirements:
            # Balanced means no single color dominates too much
            if features['max_color_frequency'] < 0.6:
                score += 1.0
            count += 1
        
        if 'diverse' in requirements:
            # Diverse means many unique colors relative to size
            diversity_ratio = features['unique_colors'] / features['total_cells']
            if diversity_ratio > 0.4:
                score += 1.0
            count += 1
        
        return (score / count if count > 0 else 0.0, 1.5)  # Medium-high weight
    
    def _evaluate_structural_requirements(self, features: Dict[str, Any], 
                                        requirements: Dict[str, Any]) -> Tuple[float, float]:
        """Evaluate structure-based requirements"""
        score = 0.0
        count = 0
        
        if 'asymmetric' in requirements:
            # Asymmetric means no major symmetries
            symmetries = [
                features['horizontal_symmetry'],
                features['vertical_symmetry'], 
                features['diagonal_symmetry']
            ]
            if not any(symmetries):
                score += 1.0
            count += 1
        
        if 'symmetric' in requirements:
            # Symmetric means has some symmetry
            symmetries = [
                features['horizontal_symmetry'],
                features['vertical_symmetry'],
                features['diagonal_symmetry']
            ]
            if any(symmetries):
                score += 1.0
            count += 1
        
        if 'high_variation' in requirements:
            if features['local_variation'] > 0.5:
                score += 1.0
            count += 1
        
        if 'predictable' in requirements:
            if features['predictability_score'] > 0.6:
                score += 1.0
            count += 1
        
        if 'complex' in requirements:
            # Complex patterns have high variation and low predictability
            complexity = features['local_variation'] * (1 - features['predictability_score'])
            if complexity > 0.3:
                score += 1.0
            count += 1
        
        return (score / count if count > 0 else 0.0, 1.0)  # Medium weight
