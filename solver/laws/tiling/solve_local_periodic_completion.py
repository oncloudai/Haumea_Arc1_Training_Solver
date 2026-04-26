import numpy as np
from typing import List, Optional

def solve_local_periodic_completion(solver) -> Optional[List[np.ndarray]]:
    for P in range(2, 6):
        for A in [-1, 0, 1]:
            for B in [-1, 0, 1]:
                if A == 0 and B == 0: continue
                consistent_task = True; results = []
                for inp, out in solver.pairs:
                    # HEURISTIC: This law as written is dangerous because it doesn't even look at inp mostly,
                    # and it returns solver.pairs[0][1].shape which is almost certainly wrong for many tasks.
                    # Let's at least check if it matches 'out' for all training pairs.
                    h, w = out.shape; mapping = {}
                    # If any 'out' has different shape than pairs[0][1], it's already suspicious
                    if out.shape != solver.pairs[0][1].shape: consistent_task = False; break
                    
                    for r in range(h):
                        for c in range(w):
                            v = (A*r + B*c) % P
                            if v in mapping:
                                if mapping[v] != out[r, c]: consistent_task = False; break
                            else: mapping[v] = out[r, c]
                        if not consistent_task: break
                    if not consistent_task: break
                if consistent_task:
                    # Verify it can reconstruct all 'out' from 'inp'
                    for inp, out in solver.pairs:
                        test_ti_map = {}
                        for r in range(inp.shape[0]):
                            for c in range(inp.shape[1]):
                                if inp[r, c] != 0: test_ti_map[(A*r + B*c) % P] = inp[r, c]
                        # This law assumes out shape is same as inp? Or constant?
                        # Task d0f5fe59 has dynamic out shape.
                        # For now, let's just fail if we can't match 'out' perfectly.
                        temp_res = np.zeros(out.shape, dtype=int)
                        for r in range(out.shape[0]):
                            for c in range(out.shape[1]):
                                temp_res[r, c] = test_ti_map.get((A*r + B*c) % P, 0)
                        if not np.array_equal(temp_res, out):
                            consistent_task = False; break
                
                if consistent_task:
                    for ti in solver.test_in:
                        ti_map = {}
                        for r in range(ti.shape[0]):
                            for c in range(ti.shape[1]):
                                if ti[r, c] != 0: ti_map[(A*r + B*c) % P] = ti[r, c]
                        
                        # Use the same shape-finding logic as in our new law if possible, 
                        # but this is tiling. Usually tiling output is same as input or fixed.
                        # For d0f5fe59, we need another law.
                        ho, wo = ti.shape # Default to ti shape
                        res = np.zeros((ho, wo), dtype=int)
                        for r in range(ho):
                            for c in range(wo): res[r, c] = ti_map.get((A*r + B*c) % P, 0)
                        results.append(res)
                    return results
    return None
