
import numpy as np
from typing import List, Optional

def solve_fit_grey_to_nearest_hole(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies grey (5) objects.
    Finds all potential 'holes' in the top area where these objects can fit.
    A hole is a set of 0-pixels matching the object shape.
    Each grey object moves to its NEAREST matching hole and turns blue (1).
    """
    def get_objects(grid, color):
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        objs = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == color and not visited[r, c]:
                    q = [(r, c)]; visited[r, c] = True
                    coords = []
                    while q:
                        cr, cc = q.pop(0)
                        coords.append((cr, cc))
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<rows and 0<=nc<cols and grid[nr,nc]==color and not visited[nr,nc]:
                                visited[nr,nc]=True; q.append((nr,nc))
                    objs.append(np.array(coords))
        return objs

    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        
        red_mask = (grid == 2)
        grey_objs = get_objects(grid, 5)
        if not grey_objs: return None
        
        output = grid.copy()
        # Clear color 5
        output[output == 5] = 0
        
        # Find all valid positions for each object shape in the top half
        # (or just anywhere that touches Red)
        for gobj in grey_objs:
            r_min, c_min = gobj.min(axis=0)
            rel_coords = gobj - [r_min, c_min]
            
            best_pos = None
            min_dist = float('inf')
            
            # Origin of gobj in input
            orig_r, orig_c = r_min, c_min
            
            # Scan potential positions
            # We only care about positions that are "up" from the original?
            # Or just any position.
            for r in range(rows):
                for c in range(cols):
                    # Check if fits
                    fit = True
                    touches_red = False
                    for dr, dc in rel_coords:
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0):
                            fit = False; break
                        # Check adjacency to Red
                        for nr2, nc2 in [(nr-1, nc), (nr+1, nc), (nr, nc-1), (nr, nc+1)]:
                            if 0 <= nr2 < rows and 0 <= nc2 < cols and grid[nr2, nc2] == 2:
                                touches_red = True
                    
                    if fit and touches_red:
                        # Calculate distance from original position
                        dist = (r - orig_r)**2 + (c - orig_c)**2
                        if dist < min_dist:
                            min_dist = dist
                            best_pos = (r, c)
            
            if best_pos:
                br, bc = best_pos
                for dr, dc in rel_coords:
                    output[br + dr, bc + dc] = 1
                    
        return output

    # Verify on pairs
    for inp, out_expected in solver.pairs:
        pred = run_single(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    test_preds = []
    for inp in solver.test_in:
        pred = run_single(inp)
        test_preds.append(pred if pred is not None else np.array(inp))
        
    return test_preds
