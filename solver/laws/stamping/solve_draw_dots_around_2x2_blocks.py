import numpy as np
from typing import List, Optional

def solve_draw_dots_around_2x2_blocks(solver) -> Optional[List[np.ndarray]]:
    """
    For each 2x2 block of color 5 with top-left (r, c), places:
    Color 1 at (r-1, c-1).
    Color 2 at (r-1, c+2).
    Color 3 at (r+2, c-1).
    Color 4 at (r+2, c+2).
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        out = grid.copy()
        
        # We should only place if it's a 2x2 block
        for r in range(h-1):
            for c in range(w-1):
                if np.all(grid[r:r+2, c:c+2] == 5):
                    # Found a 2x2 block of color 5
                    for dr, dc, color in [(-1,-1,1), (-1,2,2), (2,-1,3), (2,2,4)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            out[nr, nc] = color
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
