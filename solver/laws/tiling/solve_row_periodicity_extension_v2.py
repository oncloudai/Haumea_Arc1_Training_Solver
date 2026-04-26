
import numpy as np
from typing import List, Optional

def solve_row_periodicity_extension_v2(solver) -> Optional[List[np.ndarray]]:
    def get_period(inp):
        h = inp.shape[0]
        for P in range(1, h + 1):
            valid = True
            for i in range(h):
                if not np.array_equal(inp[i], inp[i % P]): valid = False; break
            if valid: return P
        return h

    # 1. Check if color mapping is consistent
    mapping = {}
    consistent_mapping = True
    for inp, out in solver.pairs:
        # We don't require inp.shape == out.shape here, 
        # but we need to find which pixels correspond.
        # This is tricky if shapes differ.
        # But if we assume the periodicity, we can check.
        P = get_period(inp)
        ho, wo = out.shape
        if wo != inp.shape[1]: consistent_mapping = False; break
        
        for i in range(ho):
            for j in range(wo):
                s = inp[i % P, j]
                d = out[i, j]
                if s in mapping:
                    if mapping[s] != d: consistent_mapping = False; break
                else:
                    mapping[s] = d
            if not consistent_mapping: break
        if not consistent_mapping: break
    
    if not consistent_mapping or not mapping: return None

    # 2. Check if the periodicity + mapping perfectly reconstructs all training outputs
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
        # Check if anything actually changed (to avoid identity matching everything)
        # We need a better 'any_changed' check if shapes differ.
        if inp.shape != out.shape or not np.array_equal(inp, out):
            any_changed = True
        
    if not any_changed: return None
    
    # 3. Determine output height for test cases
    # We'll use the ratio of out_h / inp_h if it's consistent, 
    # or the absolute out_h if it's constant.
    out_h_const = None
    if len(set(o.shape[0] for o in solver.train_out)) == 1:
        out_h_const = solver.train_out[0].shape[0]
    
    out_h_ratio = None
    ratios = [o.shape[0] / i.shape[0] for i, o in solver.pairs]
    if len(set(ratios)) == 1:
        out_h_ratio = ratios[0]

    results = []
    for ti in solver.test_in:
        P = get_period(ti)
        hi, wi = ti.shape
        if out_h_ratio is not None:
            ho = int(round(hi * out_h_ratio))
        elif out_h_const is not None:
            ho = out_h_const
        else:
            return None # Cannot determine output height
            
        wo = wi
        res = np.zeros((ho, wo), dtype=int)
        for i in range(ho):
            for j in range(wo):
                val = ti[i % P, j]
                res[i, j] = mapping.get(val, val)
        results.append(res)
    return results
