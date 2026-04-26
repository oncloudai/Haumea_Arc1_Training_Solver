import numpy as np
from typing import List, Optional

def solve_row_periodicity_extension(solver) -> Optional[List[np.ndarray]]:
    def get_period(inp):
        h = inp.shape[0]
        for P in range(1, h + 1):
            valid = True
            for i in range(h):
                if not np.array_equal(inp[i], inp[i % P]): valid = False; break
            if valid: return P
        return h

    mapping = {}
    consistent_mapping = True
    for inp, out in solver.pairs:
        if inp.shape != out.shape: # For now only handle same shape
            consistent_mapping = False; break
        for s, d in zip(inp.flatten(), out.flatten()):
            if s in mapping and mapping[s] != d:
                consistent_mapping = False; break
            mapping[s] = d
        if not consistent_mapping: break
    
    if not consistent_mapping: return None

    any_changed = False
    for inp, out in solver.pairs:
        P = get_period(inp)
        ho, wo = out.shape
        pred = np.zeros_like(out)
        for i in range(ho):
            for j in range(wo):
                val = inp[i % P, j]
                pred[i, j] = mapping.get(val, val)
        if not np.array_equal(pred, out): return None
        if not np.array_equal(pred, inp): any_changed = True
        
    if not any_changed: return None
    
    results = []
    for ti in solver.test_in:
        P = get_period(ti)
        # Assuming test output has same height as first train output if possible
        ho = solver.train_out[0].shape[0] if len(set(o.shape[0] for o in solver.train_out)) == 1 else ti.shape[0]
        wo = ti.shape[1]
        res = np.zeros((ho, wo), dtype=int)
        for i in range(ho):
            for j in range(wo):
                val = ti[i % P, j]
                res[i, j] = mapping.get(val, val)
        results.append(res)
    return results
