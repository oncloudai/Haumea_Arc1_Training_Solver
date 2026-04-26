import numpy as np
from typing import List, Optional

def solve_l_shape_extension_to_edge(solver) -> Optional[List[np.ndarray]]:
    """
    For each non-zero pixel at (r, c) with color X:
    1. Extend it horizontally to the right edge (W-1).
    2. Then extend it vertically downwards.
    The vertical extension length is determined by the distance to the next
    pixel's row, or the bottom of the grid if it's the last one.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        out = grid.copy()
        
        # Find all non-zero pixels and sort by row
        rows, cols = np.where(grid != 0)
        if len(rows) == 0: return None
        
        pixels = []
        for r, c in zip(rows, cols):
            pixels.append((r, c, grid[r, c]))
        pixels.sort() # Sort by row, then col
        
        for i in range(len(pixels)):
            r, c, color = pixels[i]
            # Find next pixel's row (not necessarily the same color)
            r_next = h
            for j in range(i + 1, len(pixels)):
                if pixels[j][0] > r:
                    r_next = pixels[j][0]
                    break
            
            # 1. Horizontal extension to the right
            out[r, c:w] = color
            
            # 2. Vertical extension at the right edge
            out[r + 1 : r_next, w - 1] = color
            
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
