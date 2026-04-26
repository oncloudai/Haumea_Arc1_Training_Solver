import numpy as np
from typing import List, Optional
from collections import Counter

def solve_extract_dynamic_grid_from_blocks(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a dynamic arrangement of blocks in the input.
    A block is defined by intervals of rows and columns where color transitions occur.
    The output is a grid where each cell represents a block and its dominant non-zero color.
    This generalizes extracting a 3x3 or other grid structure from larger blocks.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        # Find all transitions in rows and columns
        # A transition is where a color changes anywhere along that row/col index
        row_boundaries = {0, h}
        for r in range(1, h):
            if not np.array_equal(grid[r, :], grid[r-1, :]):
                row_boundaries.add(r)
                
        col_boundaries = {0, w}
        for c in range(1, w):
            if not np.array_equal(grid[:, c], grid[:, c-1]):
                col_boundaries.add(c)
                
        sorted_rows = sorted(list(row_boundaries))
        sorted_cols = sorted(list(col_boundaries))
        
        # Identify which intervals have non-zero content
        row_intervals = []
        for i in range(len(sorted_rows)-1):
            r_start, r_end = sorted_rows[i], sorted_rows[i+1]
            if np.any(grid[r_start:r_end, :]):
                row_intervals.append((r_start, r_end))
                
        col_intervals = []
        for j in range(len(sorted_cols)-1):
            c_start, c_end = sorted_cols[j], sorted_cols[j+1]
            if np.any(grid[:, c_start:c_end]):
                col_intervals.append((c_start, c_end))
                
        if not row_intervals or not col_intervals:
            return None
            
        out_h = len(row_intervals)
        out_w = len(col_intervals)
        out = np.zeros((out_h, out_w), dtype=int)
        
        for i, (r_start, r_end) in enumerate(row_intervals):
            for j, (c_start, c_end) in enumerate(col_intervals):
                sub = grid[r_start:r_end, c_start:c_end]
                # Dominant non-zero color
                non_zero = sub[sub != 0]
                if len(non_zero) > 0:
                    out[i, j] = Counter(non_zero).most_common(1)[0][0]
                else:
                    out[i, j] = 0
                    
        return out

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
