import numpy as np
from typing import List, Optional

def solve_propagate_first_row_pattern(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies the first non-empty row as a reference pattern.
    For every other row, maps each colored pixel to the corresponding color 
    in the reference row at the same column.
    The current row's colors then fill all columns that have their mapped 
    color in the reference row.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        # 1. Find the first non-empty row
        ref_row_idx = -1
        for r in range(h):
            if np.any(grid[r, :] != 0):
                ref_row_idx = r
                break
        if ref_row_idx == -1: return None
        
        ref_row = grid[ref_row_idx, :]
        out = np.zeros_like(grid)
        out[ref_row_idx, :] = ref_row
        
        # 2. For each other row
        for r in range(h):
            if r == ref_row_idx: continue
            if np.all(grid[r, :] == 0): continue
            
            # Mapping: Color in current row -> Color in reference row
            mapping = {} 
            for c in range(w):
                val = grid[r, c]
                if val != 0:
                    ref_val = ref_row[c]
                    mapping[val] = ref_val
            
            # Apply mapping
            for curr_c, ref_c in mapping.items():
                target_cols = np.where(ref_row == ref_c)[0]
                for tc in target_cols:
                    out[r, tc] = curr_c
                    
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
