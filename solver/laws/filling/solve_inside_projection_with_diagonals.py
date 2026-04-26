import numpy as np
from typing import List, Optional

def solve_inside_projection_with_diagonals(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies 'inside' pixels (0-pixels with 3 or 4 cardinal 'walls' of a shape color).
    Fills inside pixels with color 4.
    Projects from inside pixels in directions where there are no walls.
    Also handles diagonal projections at 'extreme' points of the inside region.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        
        unique_colors = np.unique(grid)
        shape_color = 0
        for c in unique_colors:
            if c != 0 and c != 4:
                shape_color = int(c)
                break
        
        if shape_color == 0: return grid
            
        # has_wall[r, c, d] where d is 0:Up, 1:Down, 2:Left, 3:Right
        has_wall = np.zeros((rows, cols, 4), dtype=bool)
        for r in range(rows):
            for c in range(cols):
                if np.any(grid[:r, c] == shape_color): has_wall[r, c, 0] = True
                if np.any(grid[r+1:, c] == shape_color): has_wall[r, c, 1] = True
                if np.any(grid[r, :c] == shape_color): has_wall[r, c, 2] = True
                if np.any(grid[r, c+1:] == shape_color): has_wall[r, c, 3] = True
                
        # Inside pixels I: 0-pixels with 3 or 4 walls
        inside_mask = np.zeros((rows, cols), dtype=bool)
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 0 and np.sum(has_wall[r, c]) >= 3:
                    inside_mask[r, c] = True
                    
        out = grid.copy()
        
        # Projection rules
        cardinal_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 0:Up, 1:Down, 2:Left, 3:Right
        perpendiculars = {
            0: [2, 3], # Up -> Left, Right
            1: [2, 3], # Down -> Left, Right
            2: [0, 1], # Left -> Up, Down
            3: [0, 1]  # Right -> Up, Down
        }
        
        found_any = False
        for r in range(rows):
            for c in range(cols):
                if inside_mask[r, c]:
                    found_any = True
                    out[r, c] = 4
                    for d_idx, (dr, dc) in enumerate(cardinal_dirs):
                        if not has_wall[r, c, d_idx]:
                            # 1. Project in cardinal direction D
                            pr, pc = r + dr, c + dc
                            while 0 <= pr < rows and 0 <= pc < cols:
                                if grid[pr, pc] == shape_color: break
                                out[pr, pc] = 4
                                pr += dr
                                pc += dc
                            
                            # 2. Check for extreme and project in diagonals
                            for p_idx in perpendiculars[d_idx]:
                                pdr, pdc = cardinal_dirs[p_idx]
                                nr, nc = r + pdr, c + pdc
                                is_extreme = False
                                if not (0 <= nr < rows and 0 <= nc < cols):
                                    is_extreme = True
                                elif grid[nr, nc] == shape_color:
                                    is_extreme = True
                                elif not inside_mask[nr, nc]:
                                    is_extreme = True
                                
                                if is_extreme:
                                    ddr, ddc = dr + pdr, dc + pdc
                                    pr, pc = r + ddr, c + ddc
                                    while 0 <= pr < rows and 0 <= pc < cols:
                                        if grid[pr, pc] == shape_color: break
                                        out[pr, pc] = 4
                                        pr += ddr
                                        pc += ddc
        if not found_any: return None
        return out

    for inp, out_expected in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
