
import numpy as np
from typing import List, Optional

def solve_partitioned_grid_dominant_color_mapping(solver) -> Optional[List[np.ndarray]]:
    """
    A 9x9 grid is divided into nine 3x3 blocks.
    Each block is represented by a single cell in a 3x3 output grid.
    The color of the output cell is the dominant non-zero (and non-5) color
    present in the corresponding 3x3 input block.
    """
    def apply_logic(input_grid):
        grid = np.array(input_grid)
        if grid.shape != (9, 9): return None
        out = np.zeros((3, 3), dtype=int)
        for i in range(3):
            for j in range(3):
                sub = grid[i*3:(i+1)*3, j*3:(j+1)*3]
                # Filter out 0 and 5
                colors = [c for c in sub.flatten() if c != 0 and c != 5]
                if colors:
                    dominant = max(set(colors), key=colors.count)
                    out[i, j] = dominant
        return out

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        results.append(apply_logic(ti))
    return results
