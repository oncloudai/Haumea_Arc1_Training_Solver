import numpy as np
from typing import List, Optional

def solve_propagate_shape_along_ray(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies Background, Anchor (usually color 0), and Seed (non-bg, non-anchor).
    Finds an optimal tiling vector (dr, dc) from an anchor pixel to the seed pixel.
    Propagates the shape defined by all anchor pixels along that ray.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        
        counts = np.bincount(grid.flatten(), minlength=10)
        bg = np.argmax(counts)
        anchor_pixels = [tuple(p) for p in np.argwhere(grid == 0)]
        seed_pixels = [tuple(p) for p in np.argwhere((grid != bg) & (grid != 0))]
        
        if not seed_pixels or not anchor_pixels: return grid

        seed_pos = seed_pixels[0]
        seed_color = grid[seed_pos[0], seed_pos[1]]
        anchor_set = set(anchor_pixels)
        
        candidates = []
        for p in anchor_pixels:
            dr = seed_pos[0] - p[0]
            dc = seed_pos[1] - p[1]
            if dr == 0 and dc == 0: continue
                
            overlap = False
            for a in anchor_pixels:
                if (a[0] + dr, a[1] + dc) in anchor_set:
                    overlap = True; break
            if not overlap:
                is_diag = (abs(dr) == abs(dc))
                dist = max(abs(dr), abs(dc))
                candidates.append((not is_diag, dist, dr, dc))
                
        output_grid = grid.copy()
        if candidates:
            candidates.sort()
            _, _, dr, dc = candidates[0]
            k = 1
            while True:
                points_drawn = 0
                for a in anchor_pixels:
                    nr, nc = a[0] + k * dr, a[1] + k * dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if output_grid[nr, nc] == bg:
                            output_grid[nr, nc] = seed_color
                            points_drawn += 1
                if points_drawn == 0: break
                k += 1
        return output_grid

    for inp, out in solver.pairs:
        if not np.array_equal(apply_logic(inp), out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
