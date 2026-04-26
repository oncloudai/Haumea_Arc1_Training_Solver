
import numpy as np
from typing import List, Optional

def solve_extract_closest_object_bbox_3x3(solver) -> Optional[List[np.ndarray]]:
    """
    Finds the non-zero object (connected component) closest to the color 5 pixel.
    Extracts its 3x3 bounding box as the output.
    """
    def get_objects(grid):
        grid = np.array(grid)
        labeled = np.zeros_like(grid); curr = 1
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] != 0 and grid[r, c] != 5 and labeled[r, c] == 0:
                    q = [(r, c)]; labeled[r, c] = curr
                    while q:
                        cr, cc = q.pop(0)
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] != 0 and grid[nr, nc] != 5 and labeled[nr, nc] == 0:
                                labeled[nr, nc] = curr; q.append((nr, nc))
                    curr += 1
        objs = []
        for i in range(1, curr):
            coords = np.argwhere(labeled == i)
            r_min, c_min = coords.min(axis=0); r_max, c_max = coords.max(axis=0)
            objs.append({'coords': coords, 'bbox': (r_min, c_min, r_max, c_max), 'color': grid[coords[0,0], coords[0,1]]})
        return objs

    def apply_logic(grid):
        grid = np.array(grid)
        pos5 = np.argwhere(grid == 5)
        if len(pos5) == 0: return None
        pos5 = pos5[0]
        
        objs = get_objects(grid)
        if not objs: return None
        
        best_obj = None
        min_dist = float('inf')
        for obj in objs:
            r1, c1, r2, c2 = obj['bbox']
            dist = max(0, r1 - pos5[0], pos5[0] - r2) + max(0, c1 - pos5[1], pos5[1] - c2)
            if dist < min_dist:
                min_dist = dist
                best_obj = obj
        
        if best_obj:
            r1, c1, r2, c2 = best_obj['bbox']
            # Force 3x3 extraction
            res = np.zeros((3, 3), dtype=int)
            for r, c in best_obj['coords']:
                dr, dc = r - r1, c - c1
                if 0 <= dr < 3 and 0 <= dc < 3:
                    res[dr, dc] = best_obj['color']
            return res
        return None

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
