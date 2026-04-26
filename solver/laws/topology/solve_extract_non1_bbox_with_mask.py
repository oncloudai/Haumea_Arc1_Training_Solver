import numpy as np
from typing import List, Optional

def solve_grid_a740d043(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    
    # Identify non-1 pixels
    rows, cols = np.where(grid != 1)
    
    if len(rows) == 0:
        return np.zeros((0, 0), dtype=int)
    
    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()
    
    output_grid_crop = grid[r_min:r_max+1, c_min:c_max+1]
    
    final_output = np.zeros_like(output_grid_crop)
    for r in range(output_grid_crop.shape[0]):
        for c in range(output_grid_crop.shape[1]):
            val = output_grid_crop[r, c]
            if val == 1:
                final_output[r, c] = 0
            else:
                final_output[r, c] = val
                
    return final_output

def solve_extract_non1_bbox_with_mask(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_a740d043(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_a740d043(ti) for ti in solver.test_in]
    return None
