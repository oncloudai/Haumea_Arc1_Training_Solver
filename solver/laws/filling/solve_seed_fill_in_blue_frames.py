
import numpy as np
from typing import List, Optional

def solve_seed_fill_in_blue_frames(solver) -> Optional[List[np.ndarray]]:
    """
    Find connected components of blue pixels (color 1) forming frames.
    For each frame, find a 'seed' color (not 0 or 1) within its bounding box.
    Fill all non-blue pixels in the bounding box and the row immediately above it
    with the seed color.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        out = grid.copy()
        rows, cols = grid.shape
        blue_mask = (grid == 1)
        labeled_blue = np.zeros_like(grid); curr = 1
        for r in range(rows):
            for c in range(cols):
                if blue_mask[r, c] and labeled_blue[r, c] == 0:
                    q = [(r, c)]; labeled_blue[r, c] = curr
                    while q:
                        cr, cc = q.pop(0)
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0: continue
                                nr, nc = cr + dr, cc + dc
                                if 0 <= nr < rows and 0 <= nc < cols and blue_mask[nr, nc] and labeled_blue[nr, nc] == 0:
                                    labeled_blue[nr, nc] = curr; q.append((nr, nc))
                    curr += 1
        found_any = False
        for i in range(1, curr):
            coords = np.argwhere(labeled_blue == i)
            r1, c1 = coords.min(axis=0); r2, c2 = coords.max(axis=0)
            seed = None
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    if grid[r, c] != 0 and grid[r, c] != 1:
                        seed = grid[r, c]; break
                if seed: break
            if seed:
                found_any = True
                if r1 > 0: out[r1-1, c1:c2+1] = seed
                for r in range(r1, r2 + 1):
                    for c in range(c1, c2 + 1):
                        if grid[r, c] != 1: out[r, c] = seed
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
