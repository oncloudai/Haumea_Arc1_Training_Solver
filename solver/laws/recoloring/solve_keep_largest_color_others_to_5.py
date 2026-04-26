import numpy as np
from typing import List, Optional

def solve_keep_largest_color_others_to_5(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies all colors in the input. Keeps the color with the most pixels.
    All other non-zero pixels are changed to color 5.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        colors = np.unique(grid)
        colors = [c for c in colors if c != 0]
        if not colors: return None
        
        counts = {c: np.sum(grid == c) for c in colors}
        # Find max count color. If tie, the first one found is picked.
        max_c = max(counts, key=counts.get)
        
        out = np.zeros_like(grid)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == max_c:
                    out[r, c] = max_c
                elif grid[r, c] != 0:
                    out[r, c] = 5
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
