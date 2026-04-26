
import numpy as np
from typing import List, Optional

def solve_point_reflection_around_pivot(solver) -> Optional[List[np.ndarray]]:
    """
    Finds a 'pivot' object and reflects all other non-zero pixels across its center.
    The pivot object is identified as the one that is already point-symmetric or has a unique size.
    """
    def get_objects(grid):
        grid = np.array(grid)
        labeled = np.zeros_like(grid); curr = 1
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] != 0 and labeled[r, c] == 0:
                    q = [(r, c)]; labeled[r, c] = curr
                    while q:
                        cr, cc = q.pop(0)
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<grid.shape[0] and 0<=nc<grid.shape[1] and grid[nr,nc] != 0 and labeled[nr,nc] == 0:
                                labeled[nr,nc] = curr; q.append((nr,nc))
                    curr += 1
        objs = []
        for i in range(1, curr):
            coords = np.argwhere(labeled == i)
            r_min, c_min = coords.min(axis=0); r_max, c_max = coords.max(axis=0)
            objs.append({'coords': coords, 'bbox': (r_min, c_min, r_max, c_max), 'color': grid[coords[0,0], coords[0,1]]})
        return objs

    def apply_logic(grid, pivot_obj_idx):
        grid = np.array(grid)
        objs = get_objects(grid)
        if pivot_obj_idx >= len(objs): return None
        
        pivot = objs[pivot_obj_idx]
        r1, c1, r2, c2 = pivot['bbox']
        # Pivot center (can be non-integer if we use 2*pr)
        # 2*pr = r1 + r2
        # 2*pc = c1 + c2
        pr2 = r1 + r2
        pc2 = c1 + c2
        
        out = grid.copy()
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0:
                    nr = pr2 - r
                    nc = pc2 - c
                    if 0 <= nr < h and 0 <= nc < w:
                        out[nr, nc] = grid[r, c]
        return out

    # Try each object as pivot
    for pivot_idx in range(5): # Usually 0 or 1
        all_match = True
        found_any = False
        for inp, out in solver.pairs:
            pred = apply_logic(inp, pivot_idx)
            if pred is None or not np.array_equal(pred, out):
                all_match = False; break
            found_any = True
        if all_match and found_any:
            return [apply_logic(ti, pivot_idx) for ti in solver.test_in]
            
    return None
