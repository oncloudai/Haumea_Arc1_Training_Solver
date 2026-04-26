import numpy as np
from typing import List, Optional

def solve_grid_f8c80d96(input_grid):
    grid = np.array(input_grid)
    n, m = grid.shape
    unique_colors = [c for c in np.unique(grid) if c != 0]
    
    # Heuristic for colors
    if len(unique_colors) >= 2:
        # In Example 1 and 2, both colors are in the input.
        # One is lines, one is background.
        # Usually lines have LOWER count in the input.
        counts = {c: np.sum(grid == c) for c in unique_colors}
        sorted_colors = sorted(counts.items(), key=lambda x: x[1])
        c_lines = sorted_colors[0][0]
        c_bg = sorted_colors[1][0]
    elif len(unique_colors) == 1:
        c_lines = unique_colors[0]
        c_bg = 5 # Default
    else:
        return grid

    # Prioritize centers
    potential_centers_2 = [
        (0, 0), (0, 2*(m-1)), (2*(n-1), 2*(m-1)), (2*(n-1), 0), # corners
        (-2, -2), (-2, 2*m), (2*n, 2*m), (2*n, -2), # outside
        (-1, -1), (-1, 2*m-1), (2*n-1, 2*m-1), (2*n-1, -1), # half-step outside
    ]
    # Add all integer and half-integer points in range
    for r in range(-1, n + 1):
        for c in range(-1, m + 1):
            potential_centers_2.append((2*r, 2*c))
            potential_centers_2.append((2*r+1, 2*c+1))

    # Pattern search
    for r0_2, c0_2 in potential_centers_2:
        d2_grid = np.maximum(np.abs(2*np.arange(n)[:, None] - r0_2), 
                             np.abs(2*np.arange(m)[None, :] - c0_2))
        
        mapping = {}
        possible = True
        for r in range(n):
            for c in range(m):
                if grid[r, c] != 0:
                    d2 = d2_grid[r, c]
                    if d2 in mapping and mapping[d2] != grid[r, c]:
                        possible = False; break
                    mapping[d2] = grid[r, c]
            if not possible: break
        
        if possible and mapping:
            # We must find a period S such that all mapped d2 fit.
            # And it must explain why grid[r,c] is c_lines.
            for S in range(2, 11):
                S2 = 2 * S
                # Actually, pattern can be multiple remainders.
                
                # Let's try: color is c_lines if d2 % S2 == offset, else c_bg
                for offset in range(S2):
                    match = True
                    found_any = False
                    for d2, color in mapping.items():
                        expected = c_lines if d2 % S2 == offset else c_bg
                        if color != expected:
                            match = False; break
                        found_any = True
                    
                    if match and found_any:
                        # Construct and return
                        res = np.full((n, m), c_bg, dtype=int)
                        for r in range(n):
                            for c in range(m):
                                if d2_grid[r, c] % S2 == offset:
                                    res[r, c] = c_lines
                        return res
                        
    return grid

def solve_radial_concentric_circles(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_f8c80d96(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_f8c80d96(ti) for ti in solver.test_in]
    return None
