import numpy as np
from typing import List, Optional

def solve_project_seeds_to_nearest_perpendicular_lines(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies 'seeds' (usually color 2) and 'lines' (usually color 8 rows/cols).
    Project each seed cell to the NEAREST perpendicular lines in each direction.
    Creates 3-cell wide boxes (borders) at the intersection points on the lines.
    Replaces line color with seed color at intersection points.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        # 1. Identify seeds (color 2)
        twos = [tuple(p) for p in np.argwhere(grid == 2)]
        
        # 2. Identify full lines (color 8)
        horizontal_8_lines = [r for r in range(h) if np.all(grid[r, :] == 8)]
        vertical_8_lines = [c for c in range(w) if np.all(grid[:, c] == 8)]
        
        if not twos or (not horizontal_8_lines and not vertical_8_lines):
            return grid
            
        out = grid.copy()
        
        # PHASE 1: Create 3-cell borders for nearest 8-lines only
        for two_r, two_c in twos:
            # Horizontal lines
            h_above = [r for r in horizontal_8_lines if r < two_r]
            h_below = [r for r in horizontal_8_lines if r > two_r]
            nearest_h_above = max(h_above) if h_above else None
            nearest_h_below = min(h_below) if h_below else None
            
            for h8_row in [nearest_h_above, nearest_h_below]:
                if h8_row is not None:
                    for dr in [-1, 1]:
                        row_pos = h8_row + dr
                        if 0 <= row_pos < h:
                            for dc in [-1, 0, 1]:
                                c_pos = two_c + dc
                                if 0 <= c_pos < w and out[row_pos, c_pos] == 0:
                                    out[row_pos, c_pos] = 8
            
            # Vertical lines
            v_left = [c for c in vertical_8_lines if c < two_c]
            v_right = [c for c in vertical_8_lines if c > two_c]
            nearest_v_left = max(v_left) if v_left else None
            nearest_v_right = min(v_right) if v_right else None
            
            for v8_col in [nearest_v_left, nearest_v_right]:
                if v8_col is not None:
                    for dc in [-1, 1]:
                        col_pos = v8_col + dc
                        if 0 <= col_pos < w:
                            for dr in [-1, 0, 1]:
                                r_pos = two_r + dr
                                if 0 <= r_pos < h and out[r_pos, col_pos] == 0:
                                    out[r_pos, col_pos] = 8
                                    
        # PHASE 2: Extend reds from each seed to nearest 8-lines only
        for two_r, two_c in twos:
            h_above = [r for r in horizontal_8_lines if r < two_r]
            h_below = [r for r in horizontal_8_lines if r > two_r]
            nearest_h_above = max(h_above) if h_above else None
            nearest_h_below = min(h_below) if h_below else None
            
            if nearest_h_above is not None:
                for r in range(nearest_h_above + 1, two_r):
                    if out[r, two_c] == 0: out[r, two_c] = 2
            if nearest_h_below is not None:
                for r in range(two_r + 1, nearest_h_below):
                    if out[r, two_c] == 0: out[r, two_c] = 2
                    
            v_left = [c for c in vertical_8_lines if c < two_c]
            v_right = [c for c in vertical_8_lines if c > two_c]
            nearest_v_left = max(v_left) if v_left else None
            nearest_v_right = min(v_right) if v_right else None
            
            if nearest_v_left is not None:
                for c in range(nearest_v_left + 1, two_c):
                    if out[two_r, c] == 0: out[two_r, c] = 2
            if nearest_v_right is not None:
                for c in range(two_c + 1, nearest_v_right):
                    if out[two_r, c] == 0: out[two_r, c] = 2
                    
        # PHASE 3: Place red cells at intersection points
        for two_r, two_c in twos:
            h_above = [r for r in horizontal_8_lines if r < two_r]
            h_below = [r for r in horizontal_8_lines if r > two_r]
            for h8_row in [max(h_above) if h_above else None, min(h_below) if h_below else None]:
                if h8_row is not None: out[h8_row, two_c] = 2
                
            v_left = [c for c in vertical_8_lines if c < two_c]
            v_right = [c for c in vertical_8_lines if c > two_c]
            for v8_col in [max(v_left) if v_left else None, min(v_right) if v_right else None]:
                if v8_col is not None: out[two_r, v8_col] = 2
        return out

    for inp, out in solver.pairs:
        if not np.array_equal(apply_logic(inp), out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
