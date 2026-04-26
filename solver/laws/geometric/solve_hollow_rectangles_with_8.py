
import numpy as np
from typing import List, Optional

def solve_hollow_rectangles_with_8(solver) -> Optional[List[np.ndarray]]:
    """
    Find all solid rectangular objects of a uniform color and hollow them out
    by setting their interior to 8.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        out = grid.copy()
        rows, cols = grid.shape
        labeled = np.zeros_like(grid)
        curr = 1
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0 and labeled[r, c] == 0:
                    q = [(r, c)]; labeled[r, c] = curr
                    while q:
                        cr, cc = q.pop(0)
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0: continue
                                nr, nc = cr + dr, cc + dc
                                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != 0 and labeled[nr, nc] == 0:
                                    labeled[nr, nc] = curr; q.append((nr, nc))
                    curr += 1
        
        found_any = False
        for i in range(1, curr):
            coords = np.argwhere(labeled == i)
            if len(coords) == 0: continue
            r_min, c_min = coords.min(axis=0); r_max, c_max = coords.max(axis=0)
            color = grid[coords[0,0], coords[0,1]]
            # Check if it's a solid rectangle
            if len(coords) == (r_max - r_min + 1) * (c_max - c_min + 1) and np.all(grid[r_min:r_max+1, c_min:c_max+1] == color):
                if r_max - r_min >= 2 and c_max - c_min >= 2:
                    out[r_min+1:r_max, c_min+1:c_max] = 8
                    found_any = True
        return out, found_any

    for inp, out in solver.pairs:
        pred, found = apply_logic(inp)
        if not found or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        res, _ = apply_logic(ti)
        results.append(res)
    return results
