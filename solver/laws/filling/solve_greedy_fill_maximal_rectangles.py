import numpy as np
from typing import List, Optional

def solve_grid_a8d7556c(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    
    def find_all_maximal_rectangles(g, min_h=2, min_w=2):
        hh, ww = g.shape
        rects = []
        heights = np.zeros((hh, ww), dtype=int)
        for r in range(hh):
            for c in range(ww):
                if g[r, c] == 0: heights[r, c] = heights[r-1, c] + 1 if r > 0 else 1
        for r in range(hh):
            h_row = heights[r, :]
            for c1 in range(ww):
                m_height = h_row[c1]
                for c2 in range(c1, ww):
                    m_height = min(m_height, h_row[c2])
                    if m_height < min_h: break
                    if c2 - c1 + 1 < min_w: continue
                    r1, r2, cc1, cc2 = r - m_height + 1, r, c1, c2
                    if r1 > 0 and np.all(g[r1-1, cc1:cc2+1] == 0): continue
                    if r2 < hh-1 and np.all(g[r2+1, cc1:cc2+1] == 0): continue
                    if cc1 > 0 and np.all(g[r1:r2+1, cc1-1] == 0): continue
                    if cc2 < ww-1 and np.all(g[r1:r2+1, cc2+1] == 0): continue
                    rects.append((r1, cc1, r2, cc2))
        return list(set(rects))

    rects = find_all_maximal_rectangles(grid, 2, 2)
    rects.sort(key=lambda rr: ((rr[2]-rr[0]+1)*(rr[3]-rr[1]+1), -rr[0], -rr[1]), reverse=True)
    output_grid = grid.copy()
    filled_mask = np.zeros_like(grid, dtype=bool)
    for r1, c1, r2, c2 in rects:
        if not np.any(filled_mask[r1:r2+1, c1:c2+1]):
            output_grid[r1:r2+1, c1:c2+1] = 2
            filled_mask[r1:r2+1, c1:c2+1] = True
    return output_grid

def solve_greedy_fill_maximal_rectangles(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_a8d7556c(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_a8d7556c(ti) for ti in solver.test_in]
    return None
