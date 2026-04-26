import numpy as np
from typing import List, Optional

def solve_grid_f8ff0b80(input_grid):
    grid = np.array(input_grid)
    unique_colors = np.unique(grid)
    counts = []
    for c in unique_colors:
        if c == 0: continue
        count = np.sum(grid == c)
        counts.append((c, count))
    sorted_counts = sorted(counts, key=lambda x: x[1], reverse=True)
    output = np.array([[int(c)] for c, count in sorted_counts[:3]])
    return output

def solve_extract_top_3_colors_to_3x1(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_f8ff0b80(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_f8ff0b80(ti) for ti in solver.test_in]
    return None
