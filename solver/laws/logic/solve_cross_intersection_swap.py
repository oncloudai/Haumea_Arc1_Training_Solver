import numpy as np
from typing import List, Optional

def solve_cross_intersection_swap(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        consistent = True
        found_any = False
        for inp, out in solver.pairs:
            h, w = inp.shape
            if inp.shape != out.shape: consistent = False; break
            
            unq = np.unique(inp); colors = [c for c in unq if c != bg]
            if len(colors) != 2: consistent = False; break
            c1, c2 = colors
            
            # Find all pixels of each color
            coords1 = np.argwhere(inp == c1)
            coords2 = np.argwhere(inp == c2)
            
            # Identify full row/col spans for each color
            rows1 = np.unique(coords1[:, 0]); cols1 = np.unique(coords1[:, 1])
            rows2 = np.unique(coords2[:, 0]); cols2 = np.unique(coords2[:, 1])
            
            # Potential intersection regions
            # Color 1 spans rows1, cols1. Color 2 spans rows2, cols2.
            # Intersection is where one spans row and other spans col (or vice versa)
            
            pred = inp.copy()
            # Case A: c1 is horizontal line(s), c2 is vertical line(s)
            # Intersection is (rows1, cols2)
            int_rows = rows1
            int_cols = cols2
            
            # Wait, the rule is simpler: any pixel that BELONGS to both "line spans" 
            # (one color's row span and other color's col span) is an intersection.
            
            changed = False
            for r in range(h):
                for c in range(w):
                    # Is (r, c) at intersection of c1-row and c2-col?
                    if r in rows1 and c in cols2:
                        val_in = inp[r, c]
                        val_out = c2 if val_in == c1 else c1
                        if val_in != val_out:
                            pred[r, c] = val_out; changed = True
                    # Is (r, c) at intersection of c2-row and c1-col?
                    elif r in rows2 and c in cols1:
                        val_in = inp[r, c]
                        val_out = c1 if val_in == c2 else c2
                        if val_in != val_out:
                            pred[r, c] = val_out; changed = True
            
            if not np.array_equal(pred, out):
                consistent = False; break
            if changed: found_any = True
            
        if consistent and found_any:
            def process(grid):
                h, w = grid.shape
                unq = np.unique(grid); colors = [c for c in unq if c != bg]
                if len(colors) != 2: return grid
                c1, c2 = colors
                r1 = np.unique(np.argwhere(grid == c1)[:, 0]); c1_cols = np.unique(np.argwhere(grid == c1)[:, 1])
                r2 = np.unique(np.argwhere(grid == c2)[:, 0]); c2_cols = np.unique(np.argwhere(grid == c2)[:, 1])
                res = grid.copy()
                for r in range(h):
                    for c in range(w):
                        if r in r1 and c in c2_cols:
                            res[r, c] = c2 if grid[r, c] == c1 else c1
                        elif r in r2 and c in c1_cols:
                            res[r, c] = c1 if grid[r, c] == c2 else c2
                return res
            return [process(ti) for ti in solver.test_in]
    return None
