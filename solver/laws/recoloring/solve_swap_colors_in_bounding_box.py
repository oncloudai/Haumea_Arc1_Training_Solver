import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_swap_colors_in_bounding_box(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        consistent = True
        found_any = False
        for inp, out in solver.pairs:
            # Find the bounding box of all non-bg pixels in input
            non_bg = np.argwhere(inp != bg)
            if len(non_bg) == 0: consistent = False; break
            r1, c1 = non_bg.min(axis=0)
            r2, c2 = non_bg.max(axis=0)
            sub_in = inp[r1:r2+1, c1:c2+1]
            
            if out.shape != sub_in.shape: consistent = False; break
            
            # Identify the two colors involved (other than bg)
            unq = np.unique(sub_in)
            colors = [c for c in unq if c != bg]
            if len(colors) != 2: consistent = False; break
            c_a, c_b = colors[0], colors[1]
            
            # Try swapping them
            def swap(grid, v1, v2):
                res = grid.copy()
                mask1 = (grid == v1)
                mask2 = (grid == v2)
                res[mask1] = v2
                res[mask2] = v1
                return res
            
            if np.array_equal(swap(sub_in, c_a, c_b), out):
                found_any = True
            else:
                consistent = False; break
        
        if consistent and found_any:
            results = []
            for ti in solver.test_in:
                non_bg = np.argwhere(ti != bg)
                if len(non_bg) == 0: return None
                r1, c1 = non_bg.min(axis=0); r2, c2 = non_bg.max(axis=0)
                sub_ti = ti[r1:r2+1, c1:c2+1]
                unq = np.unique(sub_ti); colors = [c for c in unq if c != bg]
                if len(colors) == 2:
                    c_a, c_b = colors[0], colors[1]
                    # Swap
                    res = sub_ti.copy()
                    res[sub_ti == c_a] = c_b
                    res[sub_ti == c_b] = c_a
                    results.append(res)
                else: return None
            return results
    return None
