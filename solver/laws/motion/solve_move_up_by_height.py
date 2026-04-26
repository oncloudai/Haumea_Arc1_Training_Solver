import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_move_up_by_height(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies all objects and moves each one UP by its own height.
    Used for task 5521c0d9.
    """
    def run_single(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        out = np.zeros_like(grid)
        
        unique_colors = np.unique(grid)
        for c in unique_colors:
            if c == 0: continue
            mask = (grid == c)
            labeled, num = label(mask)
            for i in range(1, num + 1):
                coords = np.argwhere(labeled == i)
                r_min = coords[:, 0].min()
                r_max = coords[:, 0].max()
                h = r_max - r_min + 1
                for r, c_idx in coords:
                    nr = r - h
                    if 0 <= nr < rows:
                        out[nr, c_idx] = c
        return out

    results = []
    for inp, out_expected in solver.pairs:
        pred = run_single(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    for ti in solver.test_in:
        results.append(run_single(ti))
    return results
