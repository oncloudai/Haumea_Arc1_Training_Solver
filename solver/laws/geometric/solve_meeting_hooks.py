import numpy as np
from typing import List, Optional

def solve_meeting_hooks(solver) -> Optional[List[np.ndarray]]:
    def process(grid):
        # Find all non-zero pixels
        coords = np.argwhere(grid != 0)
        if len(coords) != 2: return None
        r1, c1 = coords[0]; color1 = grid[r1, c1]
        r2, c2 = coords[1]; color2 = grid[r2, c2]
        
        res = grid.copy()
        h_grid, w_grid = grid.shape
        
        if r1 == r2: # Horizontal
            if c1 > c2: # Swap to make c1 < c2
                r1, c1, color1, r2, c2, color2 = r2, c2, color2, r1, c1, color1
            
            M = (c1 + c2) / 2.0
            # C1 hooks right, C2 hooks left
            c1_outer, c1_inner = int(M - 1.5), int(M - 0.5)
            c2_outer, c2_inner = int(M + 1.5), int(M + 0.5)
            
            # C1's hook
            for c in range(c1, c1_outer + 1):
                if 0 <= c < w_grid: res[r1, c] = color1
            for r in range(r1 - 2, r1 + 3):
                if 0 <= r < h_grid and 0 <= c1_outer < w_grid: res[r, c1_outer] = color1
            for r in [r1 - 2, r1 + 3 - 1]:
                if 0 <= r < h_grid and 0 <= c1_inner < w_grid: res[r, c1_inner] = color1
                
            # C2's hook
            for c in range(c2_outer, c2 + 1):
                if 0 <= c < w_grid: res[r2, c] = color2
            for r in range(r2 - 2, r2 + 3):
                if 0 <= r < h_grid and 0 <= c2_outer < w_grid: res[r, c2_outer] = color2
            for r in [r2 - 2, r2 + 3 - 1]:
                if 0 <= r < h_grid and 0 <= c2_inner < w_grid: res[r, c2_inner] = color2
            return res
            
        elif c1 == c2: # Vertical
            if r1 > r2:
                r1, c1, color1, r2, c2, color2 = r2, c2, color2, r1, c1, color1
            
            M = (r1 + r2) / 2.0
            r1_outer, r1_inner = int(M - 1.5), int(M - 0.5)
            r2_outer, r2_inner = int(M + 1.5), int(M + 0.5)
            
            # C1's hook
            for r in range(r1, r1_outer + 1):
                if 0 <= r < h_grid: res[r, c1] = color1
            for c in range(c1 - 2, c1 + 3):
                if 0 <= c < w_grid and 0 <= r1_outer < h_grid: res[r1_outer, c] = color1
            for c in [c1 - 2, c1 + 3 - 1]:
                if 0 <= c < w_grid and 0 <= r1_inner < h_grid: res[r1_inner, c] = color1
                
            # C2's hook
            for r in range(r2_outer, r2 + 1):
                if 0 <= r < h_grid: res[r, c2] = color2
            for c in range(c2 - 2, c2 + 3):
                if 0 <= c < w_grid and 0 <= r2_outer < h_grid: res[r2_outer, c] = color2
            for c in [c2 - 2, c2 + 3 - 1]:
                if 0 <= c < w_grid and 0 <= r2_inner < h_grid: res[r2_inner, c] = color2
            return res
            
        return None

    consistent = True
    for inp, out in solver.pairs:
        pred = process(inp)
        if pred is None or not np.array_equal(pred, out):
            consistent = False; break
    if consistent:
        results = []
        for ti in solver.test_in:
            res = process(ti)
            if res is None: return None
            results.append(res)
        return results
    return None
