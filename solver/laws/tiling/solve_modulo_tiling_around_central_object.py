import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_modulo_tiling_around_central_object(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a central color 8 object. The output size is S+2.
    The output is formed by mapping each pixel (r, c) of the input to 
    ( (r - r_offset) % size, (c - c_offset) % size ) in the output.
    """
    def get_bbox(mask):
        rows, cols = np.where(mask)
        if len(rows) == 0: return None
        return rows.min(), rows.max(), cols.min(), cols.max()

    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        bbox8 = get_bbox(grid == 8)
        if not bbox8: return None
        r1, r2, c1, c2 = bbox8
        S = r2 - r1 + 1
        # In these tasks, S_r usually equals S_c
        size = S + 2
        
        # We need to find the correct offsets.
        # Let's try all possible offsets [0...size-1]
        for dr in range(size):
            for dc in range(size):
                out = np.zeros((size, size), dtype=int)
                # Apply modulo tiling
                for r in range(h):
                    for c in range(w):
                        if grid[r, c] != 0:
                            tr = (r - dr) % size
                            tc = (c - dc) % size
                            out[tr, tc] = grid[r, c]
                
                # Check if this matches any of the training outputs?
                # (We'll do this outside the loop)
                pass
        
        # Instead of nested loops, let's derive dr, dc from the central object
        # The central object at [r1, r2, c1, c2] should map to [1, S, 1, S] in output.
        # So (r1 - dr) % size = 1  => dr = (r1 - 1) % size
        # And (c1 - dc) % size = 1 => dc = (c1 - 1) % size
        
        dr = (r1 - 1) % size
        dc = (c1 - 1) % size
        
        out = np.zeros((size, size), dtype=int)
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0:
                    tr = (r - dr) % size
                    tc = (c - dc) % size
                    # If multiple colors map to same cell, the last one wins
                    # (Usually they are disjoint in these tasks)
                    out[tr, tc] = grid[r, c]
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
