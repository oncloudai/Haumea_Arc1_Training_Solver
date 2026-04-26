import numpy as np
from typing import List, Optional

def find_symmetry_in_row(grid, r):
    if r < 0 or r >= grid.shape[0]: return None
    row = grid[r]
    unq = [c for c in np.unique(row) if c != 0]
    if not unq: return None
    
    # Try each color in the row
    for color in unq:
        cols = np.where(row == color)[0]
        if len(cols) < 2: continue
        mid = (cols.min() + cols.max()) / 2.0
        is_sym = True
        for c in cols:
            if round(2*mid - c) not in cols:
                is_sym = False; break
        if is_sym:
            return mid
    return None

def solve_reflection_by_adjacent_symmetry(solver) -> Optional[List[np.ndarray]]:
    """
    For each grid, identifies a target color block, looks at the row above it
    for a symmetry axis, and reflects the target pixels across that axis.
    """
    # We need to find which color is the 'target' color for each pair.
    # Since it might be different, we try to find a rule that identifies it.
    
    def apply_rule(grid):
        # Try each color as a candidate target
        unq = [c for c in np.unique(grid) if c != 0]
        for target_color in unq:
            coords = np.argwhere(grid == target_color)
            if coords.size == 0: continue
            min_r = coords[:,0].min()
            
            # Look at row above
            axis = find_symmetry_in_row(grid, min_r - 1)
            if axis is not None:
                # Try reflecting target_color across axis
                res = grid.copy()
                for r, c in coords:
                    c_ref = int(round(2 * axis - c))
                    if 0 <= c_ref < grid.shape[1]:
                        res[r, c] = grid[r, c_ref]
                    else:
                        res[r, c] = 0
                return res
        return None

    # Verify rule on all training pairs
    for inp, out in solver.pairs:
        pred = apply_rule(inp)
        if pred is None or not np.array_equal(pred, out):
            return None
            
    # Apply to test
    results = []
    for ti in solver.test_in:
        res = apply_rule(ti)
        if res is None: return None
        results.append(res)
    return results
