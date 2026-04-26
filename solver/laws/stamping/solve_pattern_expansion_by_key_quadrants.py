import numpy as np
from typing import List, Optional

def solve_grid_7c008303(grid):
    grid = np.array(grid)
    h, w = grid.shape
    r_line, c_line = -1, -1
    for r in range(h):
        if np.all(grid[r] == 8): r_line = r; break
    for c in range(w):
        if np.all(grid[:, c] == 8): c_line = c; break
    if r_line == -1 or c_line == -1: return grid
    quads = {(0, 0): grid[0:r_line, 0:c_line], (0, 1): grid[0:r_line, c_line+1:w],
             (1, 0): grid[r_line+1:h, 0:c_line], (1, 1): grid[r_line+1:h, c_line+1:w]}
    key_pos, pattern_pos = None, None
    for pos, q in quads.items():
        if q.shape == (2, 2): key_pos = pos
        elif q.shape == (6, 6): pattern_pos = pos
    if key_pos is None or pattern_pos is None: return grid
    key, pattern = quads[key_pos], quads[pattern_pos]
    out = np.zeros((6, 6), dtype=int)
    for kr in range(2):
        for kc in range(2):
            color = int(key[kr, kc])
            if color == 0: continue
            p_block = pattern[kr*3 : (kr+1)*3, kc*3 : (kc+1)*3]
            for pr in range(3):
                for pc in range(3):
                    if p_block[pr, pc] == 3: out[kr*3 + pr, kc*3 + pc] = color
    return out

def solve_pattern_expansion_by_key_quadrants(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_7c008303(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_7c008303(ti) for ti in solver.test_in]
    return None
