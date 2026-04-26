import numpy as np
from typing import List, Optional

def solve_map_unique_color_to_3x3_pattern(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a mapping from a unique non-zero color in the input to a fixed 3x3 pattern in the output.
    Each training example must have exactly one non-zero color in its input and a 3x3 output.
    """
    color_to_pattern = {}
    
    for inp, out in solver.pairs:
        if out.shape != (3, 3):
            return None
            
        unique_colors = np.unique(inp)
        non_zero = unique_colors[unique_colors != 0]
        
        if len(non_zero) != 1:
            return None
            
        color = int(non_zero[0])
        pattern_tuple = tuple(out.flatten().tolist())
        
        if color in color_to_pattern:
            if color_to_pattern[color] != pattern_tuple:
                return None
        else:
            color_to_pattern[color] = pattern_tuple
            
    # Apply to test inputs
    results = []
    for ti in solver.test_in:
        unique_colors = np.unique(ti)
        non_zero = unique_colors[unique_colors != 0]
        
        if len(non_zero) != 1:
            return None
            
        color = int(non_zero[0])
        if color not in color_to_pattern:
            return None
            
        pattern = np.array(color_to_pattern[color]).reshape((3, 3))
        results.append(pattern)
        
    return results
