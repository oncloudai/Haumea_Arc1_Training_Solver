
import numpy as np
from typing import List, Optional

def solve_flood_fill_partitioned_grid(solver) -> Optional[List[np.ndarray]]:
    """
    Grid is partitioned by 5-lines. Each partition has a marker C.
    The partition is flood-filled with F = C + offset.
    For 54d9e175, offset is 5.
    """
    def apply_logic(grid, offset):
        grid = np.array(grid)
        rows, cols = grid.shape
        out = grid.copy()
        
        # Find partitions
        row_seps = [r for r in range(rows) if np.all(grid[r, :] == 5)]
        col_seps = [c for c in range(cols) if np.all(grid[:, c] == 5)]
        
        # Add boundaries
        rs = [-1] + row_seps + [rows]
        cs = [-1] + col_seps + [cols]
        
        found_any = False
        for i in range(len(rs) - 1):
            for j in range(len(cs) - 1):
                r_start, r_end = rs[i] + 1, rs[i+1]
                c_start, c_end = cs[j] + 1, cs[j+1]
                
                if r_start < r_end and c_start < c_end:
                    sub = grid[r_start:r_end, c_start:c_end]
                    # Find marker (unique color other than 0 and 5)
                    unique = np.unique(sub)
                    markers = [c for c in unique if c != 0 and c != 5]
                    if len(markers) == 1:
                        marker = markers[0]
                        fill_color = marker + offset
                        out[r_start:r_end, c_start:c_end] = fill_color
                        found_any = True
        return out, found_any

    # Try different offsets, but 5 is common for this task
    for offset in [5]:
        consistent = True
        for inp, out in solver.pairs:
            pred, found = apply_logic(inp, offset)
            if not found or not np.array_equal(pred, out):
                consistent = False; break
        if consistent:
            results = []
            for ti in solver.test_in:
                res, _ = apply_logic(ti, offset)
                results.append(res)
            return results
    return None
