import numpy as np
from typing import List, Optional

def solve_recolor_in_bounding_box(solver) -> Optional[List[np.ndarray]]:
    """
    Finds a marker color, identifies its bounding box, and recolors a target color
    within that bounding box to a result color.
    """
    input_colors = []
    output_colors = []
    for inp, out in solver.pairs:
        input_colors.extend(np.unique(inp).tolist())
        output_colors.extend(np.unique(out).tolist())
    
    input_colors = sorted(list(set(input_colors)))
    output_colors = sorted(list(set(output_colors)))
    
    # Try all combinations of marker, target, result colors
    for c_marker in input_colors:
        # 0 is usually background, but let's include it if needed? 
        # Usually markers are non-zero.
        if c_marker == 0: continue 
        
        for c_target in input_colors:
            if c_target == c_marker: continue
            
            for c_result in output_colors:
                if c_result == c_target: continue
                
                consistent = True
                found_change = False
                
                for inp, out in solver.pairs:
                    if inp.shape != out.shape:
                        consistent = False; break
                    
                    coords = np.argwhere(inp == c_marker)
                    if coords.size == 0:
                        if not np.array_equal(inp, out):
                            consistent = False; break
                        continue
                    
                    min_r, min_c = coords.min(axis=0)
                    max_r, max_c = coords.max(axis=0)
                    
                    pred = inp.copy()
                    for r in range(min_r, max_r + 1):
                        for c in range(min_c, max_c + 1):
                            if pred[r, c] == c_target:
                                pred[r, c] = c_result
                                found_change = True
                    
                    if not np.array_equal(pred, out):
                        consistent = False; break
                
                if consistent and found_change:
                    # Apply to test inputs
                    test_outputs = []
                    for ti in solver.test_in:
                        coords = np.argwhere(ti == c_marker)
                        pred = ti.copy()
                        if coords.size > 0:
                            min_r, min_c = coords.min(axis=0)
                            max_r, max_c = coords.max(axis=0)
                            for r in range(min_r, max_r + 1):
                                for c in range(min_c, max_c + 1):
                                    if pred[r, c] == c_target:
                                        pred[r, c] = c_result
                        test_outputs.append(pred)
                    return test_outputs
                    
    return None
