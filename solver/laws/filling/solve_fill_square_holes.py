
import numpy as np
from typing import List, Optional

def solve_fill_square_holes(solver) -> Optional[List[np.ndarray]]:
    """
    Find all holes (zero pixels completely enclosed by non-zero pixels).
    If a hole forms a solid square, fill it with a specific color (usually 2).
    """
    def get_holes(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        bg_mask = np.zeros((rows, cols), dtype=bool)
        q = []
        for r in range(rows):
            if grid[r, 0] == 0: q.append((r, 0)); bg_mask[r, 0] = True
            if grid[r, cols-1] == 0: q.append((r, cols-1)); bg_mask[r, cols-1] = True
        for c in range(cols):
            if grid[0, c] == 0: q.append((0, c)); bg_mask[0, c] = True
            if grid[rows-1, c] == 0: q.append((rows-1, c)); bg_mask[rows-1, c] = True
        while q:
            r, c = q.pop(0)
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<rows and 0<=nc<cols and grid[nr,nc] == 0 and not bg_mask[nr,nc]:
                    bg_mask[nr,nc] = True; q.append((nr,nc))
        hole_mask = (grid == 0) & (~bg_mask)
        labeled = np.zeros_like(grid); curr = 1
        for r in range(rows):
            for c in range(cols):
                if hole_mask[r, c] and labeled[r, c] == 0:
                    q = [(r, c)]; labeled[r, c] = curr
                    while q:
                        cr, cc = q.pop(0)
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<rows and 0<=nc<cols and hole_mask[nr,nc] and labeled[nr,nc] == 0:
                                labeled[nr,nc] = curr; q.append((nr,nc))
                    curr += 1
        holes = []
        for i in range(1, curr):
            holes.append(np.argwhere(labeled == i))
        return holes

    # Try each possible fill color
    for fill_color in range(1, 10):
        consistent = True; found_any = False
        for inp, out in solver.pairs:
            res = inp.copy()
            holes = get_holes(inp)
            local_found = False
            for coords in holes:
                r_min, c_min = coords.min(axis=0); r_max, c_max = coords.max(axis=0)
                h, w = r_max - r_min + 1, c_max - c_min + 1
                if h == w and len(coords) == h * w:
                    for r, c in coords: res[r, c] = fill_color
                    local_found = True; found_any = True
            if not np.array_equal(res, out):
                consistent = False; break
        
        if consistent and found_any:
            return [solve_44d8ac46_helper(ti, fill_color, get_holes) for ti in solver.test_in]
    return None

def solve_44d8ac46_helper(grid, fill_color, get_holes_func):
    grid = np.array(grid)
    out = grid.copy()
    holes = get_holes_func(grid)
    for coords in holes:
        r_min, c_min = coords.min(axis=0); r_max, c_max = coords.max(axis=0)
        h, w = r_max - r_min + 1, c_max - c_min + 1
        if h == w and len(coords) == h * w:
            for r, c in coords: out[r, c] = fill_color
    return out
