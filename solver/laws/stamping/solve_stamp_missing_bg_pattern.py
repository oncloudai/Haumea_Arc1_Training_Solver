import numpy as np
from typing import List, Optional

def solve_grid_890034e9(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    unique_colors = [c for c in np.unique(grid) if c != 0]
    if not unique_colors: return grid
    counts = {c: int(np.sum(grid == c)) for c in unique_colors}
    sorted_colors = sorted(counts.items(), key=lambda x: x[1])
    
    # Noise/Background color is the highest count
    bg_color = sorted_colors[-1][0]
    # Frame color is the lowest count
    f_color = sorted_colors[0][0]
    
    f_coords = np.argwhere(grid == f_color)
    if len(f_coords) == 0: return grid
    fr1, fc1 = f_coords[:, 0].min(), f_coords[:, 1].min()
    fr2, fc2 = f_coords[:, 0].max(), f_coords[:, 1].max()
    fh, fw = fr2 - fr1 + 1, fc2 - fc1 + 1
    f_mask = np.zeros((fh, fw), dtype=bool)
    for r, c in f_coords:
        f_mask[r - fr1, c - fc1] = True
        
    hole_in = grid[fr1+1:fr2, fc1+1:fc2]
    hh, hw = hole_in.shape
    
    output_grid = grid.copy()
    for r in range(h - hh + 1):
        for c in range(w - hw + 1):
            if r == fr1+1 and c == fc1+1: continue
            
            region = grid[r:r+hh, c:c+hw]
            if np.all(region == 0):
                nr1, nc1 = r-1, c-1
                for ir in range(fh):
                    for ic in range(fw):
                        if f_mask[ir, ic]:
                            if 0 <= nr1 + ir < h and 0 <= nc1 + ic < w:
                                output_grid[nr1 + ir, nc1 + ic] = f_color
                                
    return output_grid

def solve_stamp_missing_bg_pattern(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_890034e9(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_890034e9(ti) for ti in solver.test_in]
    return None
