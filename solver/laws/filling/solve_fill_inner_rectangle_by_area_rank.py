
import numpy as np
from typing import List, Optional

def solve_fill_inner_rectangle_by_area_rank(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies rectangles of a specific color (4).
    Fills the inner region (excluding border) with a color based on the rectangle's area rank.
    Largest Area -> 2 (Red)
    Smallest Area -> 1 (Blue)
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
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<rows and 0<=nc<cols and grid[nr,nc]==color and not visited[nr,nc]:
                                visited[nr,nc]=True; q.append((nr,nc))
                    objs.append(np.array(coords))
        return objs

    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        
        objs = get_objects(grid, 4)
        if len(objs) < 2: return None
        
        # Calculate areas
        obj_info = []
        for obj in objs:
            r_min, c_min = obj.min(axis=0)
            r_max, c_max = obj.max(axis=0)
            h, w = r_max - r_min + 1, c_max - c_min + 1
            area = h * w
            obj_info.append({'obj': obj, 'area': area, 'bbox': (r_min, c_min, r_max, c_max)})
            
        # Sort by area
        obj_info.sort(key=lambda x: x['area'])
        
        # Smallest -> 1
        # Largest -> 2
        # What if there are more than 2?
        # The logic suggests a binary classification or rank-based.
        # Assuming only 2 for now based on examples.
        
        output = grid.copy()
        
        # Fill Smallest
        smallest = obj_info[0]
        r_min, c_min, r_max, c_max = smallest['bbox']
        # Fill inner
        for r in range(r_min + 1, r_max):
            for c in range(c_min + 1, c_max):
                output[r, c] = 1
                
        # Fill Largest
        largest = obj_info[-1]
        r_min, c_min, r_max, c_max = largest['bbox']
        for r in range(r_min + 1, r_max):
            for c in range(c_min + 1, c_max):
                output[r, c] = 2
                
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
