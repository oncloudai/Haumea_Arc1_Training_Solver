import numpy as np
from typing import List, Optional
from collections import Counter

def solve_rotational_symmetry_completion(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a center of symmetry (midpoint) among non-zero pixels of the same color.
    Completes 90-degree rotations of all non-zero pixels around that center.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        non_zero_coords = []
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0:
                    non_zero_coords.append((r, c, grid[r, c]))
                    
        if not non_zero_coords: return grid
            
        # Find the center of symmetry
        midpoints = []
        for i in range(len(non_zero_coords)):
            r1, c1, color1 = non_zero_coords[i]
            midpoints.append((float(r1), float(c1))) # A single pixel could be the center
            for j in range(i + 1, len(non_zero_coords)):
                r2, c2, color2 = non_zero_coords[j]
                if color1 == color2:
                    midpoints.append(((r1 + r2) / 2.0, (c1 + c2) / 2.0))
                    
        if not midpoints:
            center_r, center_c = (h - 1) / 2.0, (w - 1) / 2.0
        else:
            counts = Counter(midpoints)
            center_r, center_c = counts.most_common(1)[0][0]
            
        output_grid = grid.copy()
        for r, c, color in non_zero_coords:
            dr, dc = r - center_r, c - center_c
            # Rotations: (-dc, dr), (-dr, -dc), (dc, -dr)
            for nr, nc in [(-dc, dr), (-dr, -dc), (dc, -dr)]:
                abs_r = int(round(nr + center_r))
                abs_c = int(round(nc + center_c))
                if 0 <= abs_r < h and 0 <= abs_c < w:
                    output_grid[abs_r, abs_c] = color
        return output_grid

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
