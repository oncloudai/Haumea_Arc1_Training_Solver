
import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_diagonal_identity_by_object_count(solver) -> Optional[List[np.ndarray]]:
    """
    Finds the number of connected objects in the input.
    Returns a square identity matrix of that size with the object color.
    """
    consistent = True; found_any = False
    
    for inp, out in solver.pairs:
        h, w = inp.shape
        # Identify non-zero colors
        unique_colors = [c for c in np.unique(inp) if c != 0]
        if not unique_colors: consistent = False; break
        color = unique_colors[0] # Assume one non-bg color for identity
        
        mask = (inp != 0).astype(int)
        labeled, num = label(mask)
        if num == 0: consistent = False; break
        
        pred = np.zeros((num, num), dtype=int)
        for i in range(num):
            pred[i, i] = color
            
        if not np.array_equal(pred, out):
            consistent = False; break
        found_any = True
            
    if consistent and found_any:
        results = []
        for ti in solver.test_in:
            unique_colors = [c for c in np.unique(ti) if c != 0]
            color = unique_colors[0] if unique_colors else 8
            mask = (ti != 0).astype(int)
            labeled, num = label(mask)
            
            res = np.zeros((num, num), dtype=int)
            for i in range(num):
                res[i, i] = color
            results.append(res)
        return results
    return None
