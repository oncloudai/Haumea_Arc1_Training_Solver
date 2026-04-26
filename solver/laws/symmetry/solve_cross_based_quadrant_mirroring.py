
import numpy as np
from typing import List, Optional

def solve_cross_based_quadrant_mirroring(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a cross (a full row and column of the same color).
    Extracts the top-left quadrant defined by the cross.
    Recolors non-zero pixels in the quadrant to the cross color.
    Creates a 4-fold mirrored output from this quadrant.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        best_color = -1
        best_row = -1
        best_col = -1
        for color in range(1, 10):
            row_indices = [r for r in range(rows) if np.all(grid[r, :] == color)]
            col_indices = [c for c in range(cols) if np.all(grid[:, c] == color)]
            if len(row_indices) == 1 and len(col_indices) == 1:
                best_color = color
                best_row = row_indices[0]
                best_col = col_indices[0]
                break
        if best_color == -1: return None
        tl = grid[0:best_row, 0:best_col].copy()
        tl[tl != 0] = best_color
        top = np.concatenate([tl, tl[:, ::-1]], axis=1)
        bottom = np.concatenate([tl[::-1, :], tl[::-1, ::-1]], axis=1)
        return np.concatenate([top, bottom], axis=0)

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
