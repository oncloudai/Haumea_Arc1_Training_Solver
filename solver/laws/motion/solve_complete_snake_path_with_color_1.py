import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_complete_snake_path_with_color_1(solver) -> Optional[List[np.ndarray]]:
    """
    Complete snake-like paths of color 8 with color 1 to full width.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        out = grid.copy()
        
        mask = (grid == 8).astype(int)
        labeled, num_f = label(mask, structure=np.ones((3,3)))
        if num_f == 0: return None
        
        # Get all components and sort them by r_min
        components = []
        for i in range(1, num_f + 1):
            rows, cols = np.where(labeled == i)
            components.append({
                'r_min': rows.min(), 'r_max': rows.max(),
                'cols': set(cols), 'pixels': list(zip(rows, cols))
            })
        components.sort(key=lambda x: x['r_min'])
        
        ref_h = components[0]['r_max'] - components[0]['r_min'] + 1
        
        for comp in components:
            r_min, r_max = comp['r_min'], comp['r_max']
            u_cols = comp['cols']
            remaining_cols = [c for c in range(w) if c not in u_cols]
            if not remaining_cols: continue
            
            # Check if sparse
            is_sparse = False
            for r in range(r_min, r_max + 1):
                if np.any(grid[r, min(u_cols):max(u_cols)+1] == 0):
                    is_sparse = True; break
            
            if is_sparse:
                for c in remaining_cols:
                    for r in range(r_min, r_max + 1):
                        row_8s = [cc for cc in range(w) if grid[r, cc] == 8]
                        if row_8s and (c % 2 == row_8s[0] % 2):
                            out[r, c] = 1
                            break
            else:
                target_r = r_min + ref_h
                if target_r < h:
                    for c in remaining_cols:
                        out[target_r, c] = 1
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
