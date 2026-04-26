import numpy as np
from typing import List, Optional

def solve_reflected_rays_from_source_through_lens(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a source block and a lens structure (two different colors stacked).
    Generates rays from the source through the lens, propagating diagonally.
    In task b8cdaf2b: source_color and lens_color are identified by adjacency.
    """
    def apply_logic(input_grid):
        input_grid = np.array(input_grid)
        rows, cols = input_grid.shape
        output_grid = np.copy(input_grid)
        
        source_color = -1
        lens_color = -1
        r_base = -1
        
        # Find source and lens colors
        # Look for a non-zero pixel with a different non-zero pixel directly above it
        for r in range(1, rows):
            for c in range(cols):
                if input_grid[r, c] != 0 and input_grid[r-1, c] != 0 and input_grid[r, c] != input_grid[r-1, c]:
                    source_color = input_grid[r, c]
                    lens_color = input_grid[r-1, c]
                    r_base = r
                    break
            if source_color != -1: break
                
        if source_color == -1: return None
            
        # Find the extent of the source block on r_base that has the lens color above it
        c_indices = []
        for c in range(cols):
            if input_grid[r_base, c] == source_color and input_grid[r_base-1, c] == lens_color:
                c_indices.append(c)
                
        if not c_indices: return None
            
        c_min = min(c_indices)
        c_max = max(c_indices)
        
        # Generate left ray from the leftmost source pixel
        k = 1
        while True:
            r_curr = r_base - 1 - k
            c_curr = c_min - k
            if 0 <= r_curr < rows and 0 <= c_curr < cols:
                output_grid[r_curr, c_curr] = source_color
                k += 1
            else: break
                
        # Generate right ray from the rightmost source pixel
        k = 1
        while True:
            r_curr = r_base - 1 - k
            c_curr = c_max + k
            if 0 <= r_curr < rows and 0 <= c_curr < cols:
                output_grid[r_curr, c_curr] = source_color
                k += 1
            else: break
                
        return output_grid

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
