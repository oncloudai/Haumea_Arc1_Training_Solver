import numpy as np
from typing import List, Optional

def solve_cast_rays_from_source_through_internal_corners(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a 'frame' color (heuristic: the color with the fewest internal L-corners).
    Identifies 'internal corners' of that frame.
    Identifies 'source' colors (other non-zero colors).
    Casts diagonal rays from source pixels through the corners until hitting the frame.
    """
    def get_internal_corners(grid, color):
        rows, cols = grid.shape
        corners = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == color:
                    has_v = (r > 0 and grid[r-1, c] == color) or (r < rows-1 and grid[r+1, c] == color)
                    has_h = (c > 0 and grid[r, c-1] == color) or (c < cols-1 and grid[r, c+1] == color)
                    if has_v and has_h:
                        corners.append((r, c))
        return corners

    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        unique_colors = [int(c) for c in np.unique(grid) if c != 0]
        if len(unique_colors) < 2: return grid

        candidates = []
        for color in unique_colors:
            corners = get_internal_corners(grid, color)
            if corners: candidates.append((color, corners))
        
        if not candidates: return grid
        candidates.sort(key=lambda x: len(x[1]))
        frame_color, corners = candidates[0]
        source_colors = [c for c in unique_colors if c != frame_color]
        
        out = grid.copy()
        for sr, sc in np.argwhere(np.isin(grid, source_colors)):
            source_color = grid[sr, sc]
            for cr, cc in corners:
                dr, dc = cr - sr, cc - sc
                if abs(dr) == abs(dc) and dr != 0:
                    step_r, step_c = np.sign(dr), np.sign(dc)
                    tr, tc = cr + step_r, cc + step_c
                    while 0 <= tr < rows and 0 <= tc < cols:
                        if out[tr, tc] == 0:
                            out[tr, tc] = source_color
                        elif out[tr, tc] == frame_color:
                            break
                        else:
                            if out[tr, tc] != source_color: break
                        tr += step_r
                        tc += step_c
        return out

    for inp, out in solver.pairs:
        if not np.array_equal(apply_logic(inp), out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
