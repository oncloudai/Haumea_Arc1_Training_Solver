import numpy as np
from typing import List, Optional

def is_symmetric_h(grid):
    return np.array_equal(grid, np.fliplr(grid))

def get_objects_by_color(grid):
    grid = np.array(grid)
    h, w = grid.shape
    colors = np.unique(grid)
    colors = colors[colors > 0]
    
    objects = []
    for c in colors:
        coords = np.argwhere(grid == c)
        if coords.size == 0: continue
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        
        obj_h = r_max - r_min + 1
        obj_w = c_max - c_min + 1
        obj_grid = np.zeros((obj_h, obj_w), dtype=int)
        for r, c_pos in coords:
            obj_grid[r - r_min, c_pos - c_min] = int(grid[r, c_pos])
        
        objects.append(obj_grid)
    return objects

def solve_extract_symmetric_object(solver) -> Optional[List[np.ndarray]]:
    # Task 72ca375d
    consistent = True
    for inp, out in solver.pairs:
        objects = get_objects_by_color(inp)
        found = False
        for obj in objects:
            if is_symmetric_h(obj) and np.array_equal(obj, out):
                found = True
                break
        if not found:
            consistent = False; break
            
    if consistent:
        results = []
        for ti in solver.test_in:
            objects = get_objects_by_color(ti)
            symmetric = [obj for obj in objects if is_symmetric_h(obj)]
            if symmetric:
                results.append(symmetric[0])
            else:
                # Fallback or None? MegaSolver expects a prediction.
                # If we don't return here, it might crash if solver expects fixed length list.
                return None
        return results
    return None
