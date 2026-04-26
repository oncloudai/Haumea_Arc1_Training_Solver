import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_recolor_blobs_by_size_min_vs_others(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies connected components of a single color.
    Finds the minimum size among these components.
    Recolors components of minimum size to color 2, and all other components to color 1.
    """
    consistent = True; found_any = False
    working_conn = None
    
    for inp, out in solver.pairs:
        non_zero_colors = [c for c in np.unique(inp) if c != 0]
        if len(non_zero_colors) != 1: 
            consistent = False; break
        
        match = False
        for conn in [8, 4]:
            blobs = get_blobs(inp, 0, conn)
            if not blobs: continue
            
            sizes = [len(b['coords']) for b in blobs]
            min_size = min(sizes)
            
            pred = np.zeros_like(inp)
            for b in blobs:
                color = 2 if len(b['coords']) == min_size else 1
                for r, c in b['coords']:
                    pred[r, c] = color
            
            if np.array_equal(pred, out):
                match = True
                working_conn = conn
                break
        
        if not match:
            consistent = False; break
        found_any = True

    if consistent and found_any:
        results = []
        for ti in solver.test_in:
            res = np.zeros_like(ti)
            blobs = get_blobs(ti, 0, working_conn)
            if blobs:
                sizes = [len(b['coords']) for b in blobs]
                min_size = min(sizes)
                for b in blobs:
                    color = 2 if len(b['coords']) == min_size else 1
                    for r, c in b['coords']:
                        res[r, c] = color
            results.append(res)
        return results
    return None
