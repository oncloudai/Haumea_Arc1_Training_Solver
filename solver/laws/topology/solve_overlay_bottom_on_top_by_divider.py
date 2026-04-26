import numpy as np
from typing import List, Optional

def solve_overlay_bottom_on_top_by_divider(solver) -> Optional[List[np.ndarray]]:
    """
    Finds a horizontal divider row (solid color 5).
    Splits the grid into top and bottom sections relative to this divider.
    Overlays the non-zero pixels of the bottom section onto the top section.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        # Find horizontal divider row (color 5)
        divider_rows = np.where(np.all(grid == 5, axis=1))[0]
        if len(divider_rows) == 0: return None
        
        div_idx = divider_rows[0]
        top_section = grid[:div_idx, :]
        bottom_section = grid[div_idx+1:, :]
        
        out = top_section.copy()
        rows = min(top_section.shape[0], bottom_section.shape[0])
        cols = top_section.shape[1]
        for r in range(rows):
            for c in range(cols):
                if bottom_section[r, c] != 0:
                    out[r, c] = bottom_section[r, c]
        return out

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
