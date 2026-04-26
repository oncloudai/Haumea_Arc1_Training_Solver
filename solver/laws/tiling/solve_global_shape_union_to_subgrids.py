
import numpy as np
from solver.utils import get_subgrids_by_bg

def solve_global_shape_union_to_subgrids(solver):
    # Identify delimiter color
    def apply_logic(inp):
        inp = np.array(inp)
        h, w = inp.shape
        delimiter = None
        for color in range(1, 10):
            if np.any(np.all(inp == color, axis=0)) or np.any(np.all(inp == color, axis=1)):
                delimiter = color; break
        if delimiter is None: return None
        
        subs = get_subgrids_by_bg(inp, delimiter)
        if not subs: return None
        
        # All subgrids must have same shape
        sh, sw = subs[0]['h'], subs[0]['w']
        if not all(s['h'] == sh and s['w'] == sw for s in subs): return None
        
        union_mask = np.zeros((sh, sw), dtype=bool)
        for s in subs:
            union_mask |= (s['grid'] != 0)
            
        outp = inp.copy()
        for s in subs:
            r1, r2, c1, c2 = s['r'], s['r']+s['h'], s['c'], s['c']+s['w']
            sub_out = outp[r1:r2, c1:c2].copy()
            for dr in range(sh):
                for dc in range(sw):
                    if union_mask[dr, dc] and sub_out[dr, dc] == 0:
                        sub_out[dr, dc] = delimiter
            outp[r1:r2, c1:c2] = sub_out
        return outp

    results = []
    for inp, outp in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, outp):
            return None
            
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
