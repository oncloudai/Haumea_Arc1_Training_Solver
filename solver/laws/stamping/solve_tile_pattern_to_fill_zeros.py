import numpy as np
from typing import List, Optional

def solve_grid_f9012d9b(input_grid):
    grid = np.array(input_grid)
    rows, cols = grid.shape
    best_h, best_w = rows, cols
    found_pattern = None
    for h in range(1, rows + 1):
        for w in range(1, cols + 1):
            pattern = np.zeros((h, w), dtype=int)
            consistent = True
            for r in range(rows):
                for c in range(cols):
                    if grid[r, c] != 0:
                        pr, pc = r % h, c % w
                        if pattern[pr, pc] == 0: pattern[pr, pc] = grid[r, c]
                        elif pattern[pr, pc] != grid[r, c]: consistent = False; break
                if not consistent: break
            if consistent and np.all(pattern != 0):
                best_h, best_w = h, w
                found_pattern = pattern
                break
        if found_pattern is not None: break
    if found_pattern is None: return np.zeros((0, 0), dtype=int)
    zero_coords = np.argwhere(grid == 0)
    if len(zero_coords) == 0: return np.zeros((0, 0), dtype=int)
    r_min, r_max = zero_coords[:, 0].min(), zero_coords[:, 0].max()
    c_min, c_max = zero_coords[:, 1].min(), zero_coords[:, 1].max()
    out_h, out_w = r_max - r_min + 1, c_max - c_min + 1
    output = np.zeros((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            actual_r = r + r_min; actual_c = c + c_min
            output[r, c] = found_pattern[actual_r % best_h, actual_c % best_w]
    return output

def solve_tile_pattern_to_fill_zeros(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_f9012d9b(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_f9012d9b(ti) for ti in solver.test_in]
    return None
