import numpy as np
from typing import List, Optional

def solve_pixel_upscaling_by_unique_colors_count(solver) -> Optional[List[np.ndarray]]:
    """
    Upscales the input grid by a factor N, where N is the number of unique non-zero colors 
    present in the input grid.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        in_rows, in_cols = grid.shape
        
        num_colors = len(set(np.unique(grid)) - {0})
        if num_colors == 0: return grid
        
        factor = num_colors
        out_rows, out_cols = in_rows * factor, in_cols * factor
        output = np.zeros((out_rows, out_cols), dtype=int)
        
        for r in range(in_rows):
            for c in range(in_cols):
                output[r*factor:(r+1)*factor, c*factor:(c+1)*factor] = grid[r, c]
        return output

    for inp, out_expected in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
