import numpy as np
from typing import List, Optional

def solve_grid_780d0b14(grid):
    grid = np.array(grid)
    row_is_0 = np.all(grid == 0, axis=1)
    col_is_0 = np.all(grid == 0, axis=0)
    
    def get_segments(is_zero):
        segments = []
        start = None
        for i, val in enumerate(is_zero):
            if not val:
                if start is None: start = i
            else:
                if start is not None:
                    segments.append((start, i - 1))
                    start = None
        if start is not None: segments.append((start, len(is_zero) - 1))
        return segments

    row_segments = get_segments(row_is_0)
    col_segments = get_segments(col_is_0)
    if not row_segments or not col_segments: return grid
    
    out_h, out_w = len(row_segments), len(col_segments)
    output = np.zeros((out_h, out_w), dtype=int)
    for r, (r_start, r_end) in enumerate(row_segments):
        for c, (c_start, c_end) in enumerate(col_segments):
            cell = grid[r_start:r_end+1, c_start:c_end+1]
            unique, counts = np.unique(cell, return_counts=True)
            mask = unique > 0
            unique, counts = unique[mask], counts[mask]
            if unique.size > 0:
                output[r, c] = unique[np.argmax(counts)]
    return output

def solve_grid_cells_to_modal_color(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_780d0b14(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_780d0b14(ti) for ti in solver.test_in]
    return None
