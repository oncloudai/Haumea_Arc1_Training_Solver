import numpy as np
from typing import List, Optional

def solve_propagate_color_along_modulo_grid(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a pivot (pr, pc) such that all non-zero pixels are on row pr or col pc.
    Propagates colors from those pixels along the row or column, 
    with a step equal to the number of unique colors T.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        non_zero_coords = np.argwhere(grid != 0)
        if len(non_zero_coords) == 0: return grid
            
        unique_colors = np.unique(grid[grid != 0])
        t = len(unique_colors)
        
        # Find pivot (pr, pc)
        pivot_r, pivot_pc = -1, -1
        found = False
        for pr in range(h):
            for pc in range(w):
                possible = True
                for r, c in non_zero_coords:
                    if r != pr and c != pc:
                        possible = False
                        break
                if possible:
                    pivot_r, pivot_pc = pr, pc
                    found = True
                    break
            if found: break
                
        if not found: return grid
            
        output_grid = np.zeros_like(grid)
        for r, c in non_zero_coords:
            color = grid[r, c]
            if r == pivot_r:
                # Propagate along the row
                for c_prime in range(w):
                    if c_prime % t == c % t:
                        output_grid[pivot_r, c_prime] = color
            if c == pivot_pc:
                # Propagate along the column
                for r_prime in range(h):
                    if r_prime % t == r % t:
                        output_grid[r_prime, pivot_pc] = color
        return output_grid

    for inp, out in solver.pairs:
        if not np.array_equal(apply_logic(inp), out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
