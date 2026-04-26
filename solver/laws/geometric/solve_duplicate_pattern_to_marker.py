import numpy as np
from typing import List, Optional

def solve_duplicate_pattern_to_marker(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a marker pixel (color 5) and a pattern of other colored pixels.
    Calculates the vector V from the center of the pattern's bounding box to the marker.
    The output is the original pattern plus a copy of the pattern shifted by V.
    Color 5 is removed.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        # 1. Find marker (color 5)
        m_rows, m_cols = np.where(grid == 5)
        if len(m_rows) == 0: return None
        mr, mc = m_rows[0], m_cols[0]
        
        # 2. Find all other non-zero pixels
        other_rows, other_cols = np.where((grid != 0) & (grid != 5))
        if len(other_rows) == 0: return None
        
        r1, r2 = other_rows.min(), other_rows.max()
        c1, c2 = other_cols.min(), other_cols.max()
        
        # Center of bbox
        center_r = (r1 + r2) // 2
        center_c = (c1 + c2) // 2
        
        # Shift vector
        dr = mr - center_r
        dc = mc - center_c
        
        # 3. Create output
        out = np.zeros_like(grid)
        # Copy original
        for r, c in zip(other_rows, other_cols):
            out[r, c] = grid[r, c]
            
        # Copy shifted
        for r, c in zip(other_rows, other_cols):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = grid[r, c]
                
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
