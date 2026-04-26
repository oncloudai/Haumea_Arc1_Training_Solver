import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_extend_color_8_with_color_1(solver) -> Optional[List[np.ndarray]]:
    """
    Finds components of color 8 and extends them with color 1.
    Handles sparse (checkerboard) components by extending each row's pattern.
    Handles solid components by filling the next row or repeating rows
    to complete a full-width path (snake logic).
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        out = grid.copy()
        
        mask = (grid == 8).astype(int)
        labeled, num_features = label(mask, structure=np.ones((3,3)))
        if num_features == 0: return None
        
        for i in range(1, num_features + 1):
            comp_mask = (labeled == i)
            rows_idx, cols_idx = np.where(comp_mask)
            r_min, r_max = rows_idx.min(), rows_idx.max()
            
            # Check if sparse
            is_sparse = False
            u_cols = set()
            comp_rows = []
            for r in range(r_min, r_max + 1):
                row_8s = np.where((grid[r, :] == 8) & comp_mask[r, :])[0]
                if len(row_8s) > 0:
                    comp_rows.append((r, row_8s))
                    for c in row_8s: u_cols.add(c)
                    if len(row_8s) < (row_8s.max() - row_8s.min() + 1):
                        is_sparse = True
            
            if is_sparse:
                # Sparse logic: extend patterns in same rows
                for r, row_8s in comp_rows:
                    p = 2
                    offset = row_8s[0] % p
                    for c in range(w):
                        if grid[r, c] == 0 and c % p == offset:
                            out[r, c] = 1
            else:
                # Solid logic: path completion
                missing = [c for c in range(w) if c not in u_cols]
                if not missing: continue
                
                if len(comp_rows) == 1:
                    # Single row: fill same row
                    for c in missing: out[r_min, c] = 1
                else:
                    # Multi-row: Snake or Next-row
                    # Check for repeating patterns (like Test 0)
                    is_repeating = False
                    if len(comp_rows) >= 3:
                        r_first, r_last = comp_rows[0][0], comp_rows[-1][0]
                        if np.array_equal(grid[r_first, :] == 8, grid[r_last, :] == 8):
                            is_repeating = True
                    
                    if is_repeating:
                        first_pattern = (grid[comp_rows[0][0], :] == 8)
                        for r, _ in comp_rows:
                            if np.array_equal(grid[r, :] == 8, first_pattern):
                                for c in missing: out[r, c] = 1
                    else:
                        # Standard solid path: fill next row
                        if r_max + 1 < h:
                            for c in missing: out[r_max + 1, c] = 1
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
