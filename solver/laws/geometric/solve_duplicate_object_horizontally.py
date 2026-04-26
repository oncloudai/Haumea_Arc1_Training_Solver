
import numpy as np
from typing import List, Optional

def get_obj(grid):
    coords = np.argwhere(grid != 0)
    if len(coords) == 0: return None
    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    return grid[min_r:max_r+1, min_c:max_c+1]

def solve_duplicate_object_horizontally(solver) -> Optional[List[np.ndarray]]:
    """
    Finds the object in the input and tiles it (tr, tc) times to match the output.
    """
    for tr in range(1, 4):
        for tc in range(1, 4):
            if tr == 1 and tc == 1: continue
            consistent = True
            for inp, out in solver.pairs:
                obj = get_obj(inp)
                if obj is None:
                    consistent = False; break
                tiled = np.tile(obj, (tr, tc))
                if tiled.shape != out.shape or not np.array_equal(tiled, out):
                    consistent = False; break
            
            if consistent:
                results = []
                for ti in solver.test_in:
                    obj = get_obj(ti)
                    if obj is None: return None
                    results.append(np.tile(obj, (tr, tc)))
                return results
    return None
