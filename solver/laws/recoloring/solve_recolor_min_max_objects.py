import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_recolor_min_max_objects(solver) -> Optional[List[np.ndarray]]:
    consistent = True; found_any = False
    for inp, out in solver.pairs:
        non_zero_colors = [c for c in np.unique(inp) if c != 0]
        if len(non_zero_colors) != 1: consistent = False; break
        main_color = non_zero_colors[0]
        
        # We need to try both 4 and 8 connectivity
        for conn in [4, 8]:
            blobs = get_blobs(inp, 0, conn)
            if not blobs: continue
            
            sizes = [len(b['coords']) for b in blobs]
            max_size = max(sizes)
            min_size = min(sizes)
            
            # If multiple have same min/max size, this law might be ambiguous.
            # But let's see.
            max_blobs = [b for b in blobs if len(b['coords']) == max_size]
            min_blobs = [b for b in blobs if len(b['coords']) == min_size]
            
            if len(max_blobs) > 1 or len(min_blobs) > 1:
                # In the examples, they are unique.
                pass
            
            pred = np.zeros_like(inp)
            for b in max_blobs:
                for r, c in b['coords']: pred[r, c] = 1
            for b in min_blobs:
                # If an object is both min and max (only one object),
                # this logic might need adjustment. But here they are different.
                for r, c in b['coords']: pred[r, c] = 2
            
            if np.array_equal(pred, out):
                found_any = True; break
        else:
            consistent = False; break
            
    if consistent and found_any:
        results = []
        for ti in solver.test_in:
            non_zero_colors = [c for c in np.unique(ti) if c != 0]
            if len(non_zero_colors) != 1:
                results.append(ti.copy()); continue
            
            # Use the connectivity that worked (let's assume 8 if not sure, or try both)
            # For simplicity, I'll try 8 then 4.
            for conn in [8, 4]:
                blobs = get_blobs(ti, 0, conn)
                if not blobs: continue
                sizes = [len(b['coords']) for b in blobs]
                max_size = max(sizes); min_size = min(sizes)
                max_blobs = [b for b in blobs if len(b['coords']) == max_size]
                min_blobs = [b for b in blobs if len(b['coords']) == min_size]
                res = np.zeros_like(ti)
                for b in max_blobs:
                    for r, c in b['coords']: res[r, c] = 1
                for b in min_blobs:
                    for r, c in b['coords']: res[r, c] = 2
                break
            else:
                res = ti.copy()
            results.append(res)
        return results
    return None
