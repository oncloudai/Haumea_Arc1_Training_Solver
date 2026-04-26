import numpy as np
from typing import List, Optional

def solve_bouncing_diagonal(solver) -> Optional[List[np.ndarray]]:
    """
    Project a diagonal path from an end of an azure (8) line.
    The path fills empty (0) cells with green (3).
    The path reflects once off a red (2) wall and stops at the grid edges.
    """
    def run_single(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        azure_coords = np.argwhere(grid == 8)
        if len(azure_coords) < 2: return None
        
        # Find ends of the azure line
        ends = []
        for r, c in azure_coords:
            adj = 0
            neighbor = None
            for ar, ac in azure_coords:
                if abs(r-ar) == 1 and abs(c-ac) == 1:
                    adj += 1
                    neighbor = (ar, ac)
            if adj == 1:
                ends.append(((r, c), (r - neighbor[0], c - neighbor[1])))
        if not ends: return None
        
        for (start_r, start_c), (vdr, vdc) in ends:
            cr, cc = start_r + vdr, start_c + vdc
            cdr, cdc = vdr, vdc
            if not (0 <= cr < rows and 0 <= cc < cols and grid[cr, cc] == 0):
                continue
            
            out = grid.copy()
            has_reflected = False
            path_found = False
            
            while 0 <= cr < rows and 0 <= cc < cols and grid[cr, cc] == 0:
                out[cr, cc] = 3
                path_found = True
                
                nr, nc = cr + cdr, cc + cdc
                
                # Reflect off Red (2)
                hit_red_r = (0 <= nr < rows and grid[nr, cc] == 2)
                hit_red_c = (0 <= nc < cols and grid[cr, nc] == 2)
                
                if hit_red_r or hit_red_c:
                    if hit_red_r: cdr = -cdr
                    if hit_red_c: cdc = -cdc
                    has_reflected = True
                    cr += cdr
                    cc += cdc
                    continue
                
                # Stop at grid edge
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    break
                
                cr, cc = nr, nc
                
            if path_found:
                return out
        return None

    # Verification
    for inp, out_expected in solver.pairs:
        pred = run_single(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    return [run_single(ti) for ti in solver.test_in]
