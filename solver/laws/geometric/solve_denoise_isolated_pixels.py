
import numpy as np
from typing import List, Optional

def solve_denoise_isolated_pixels(solver) -> Optional[List[np.ndarray]]:
    """
    Remove all non-zero pixels that have zero neighbors (8-connectivity).
    """
    def apply_logic(grid):
        grid = np.array(grid)
        out = grid.copy()
        rows, cols = grid.shape
        found_any = False
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0:
                    neighs = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != 0:
                                neighs += 1
                    if neighs == 0:
                        out[r, c] = 0
                        found_any = True
        return out, found_any

    for inp, out in solver.pairs:
        pred, found = apply_logic(inp)
        if not found or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        res, _ = apply_logic(ti)
        results.append(res)
    return results
