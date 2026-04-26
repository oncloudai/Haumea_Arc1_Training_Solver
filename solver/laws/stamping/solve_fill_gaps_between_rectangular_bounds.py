import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_fill_gaps_between_rectangular_bounds(solver) -> Optional[List[np.ndarray]]:
    fill_color = 3
    for conn in [1, 2]: # 1 is 4-connectivity, 2 is 8-connectivity
        structure = np.ones((3,3)) if conn == 2 else np.array([[0,1,0],[1,1,1],[0,1,0]])
        consistent = True; found_any = False
        
        for inp, out in solver.pairs:
            h, w = inp.shape
            labeled, num = label(inp != 0, structure=structure)
            mask = np.zeros((h, w), dtype=bool)
            for i in range(1, num + 1):
                coords = np.argwhere(labeled == i)
                r_min, c_min = coords.min(axis=0)
                r_max, c_max = coords.max(axis=0)
                mask[r_min:r_max+1, c_min:c_max+1] = True
                
            dilated = np.zeros((h, w), dtype=bool)
            for r in range(h):
                for c in range(w):
                    if mask[r, c]:
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < h and 0 <= nc < w:
                                    dilated[nr, nc] = True
            
            pred = inp.copy()
            for r in range(h):
                for c in range(w):
                    if inp[r, c] == 0 and not dilated[r, c]:
                        pred[r, c] = fill_color
                        found_any = True
                        
            if not np.array_equal(pred, out):
                consistent = False; break
                
        if consistent and found_any:
            results = []
            for ti in solver.test_in:
                h, w = ti.shape
                labeled, num = label(ti != 0, structure=structure)
                mask = np.zeros((h, w), dtype=bool)
                for i in range(1, num + 1):
                    coords = np.argwhere(labeled == i)
                    r_min, c_min = coords.min(axis=0)
                    r_max, c_max = coords.max(axis=0)
                    mask[r_min:r_max+1, c_min:c_max+1] = True
                
                dilated = np.zeros((h, w), dtype=bool)
                for r in range(h):
                    for c in range(w):
                        if mask[r, c]:
                            for dr in [-1, 0, 1]:
                                for dc in [-1, 0, 1]:
                                    nr, nc = r + dr, c + dc
                                    if 0 <= nr < h and 0 <= nc < w: dilated[nr, nc] = True
                
                res = ti.copy()
                for r in range(h):
                    for c in range(w):
                        if ti[r, c] == 0 and not dilated[r, c]:
                            res[r, c] = fill_color
                results.append(res)
            return results
    return None
