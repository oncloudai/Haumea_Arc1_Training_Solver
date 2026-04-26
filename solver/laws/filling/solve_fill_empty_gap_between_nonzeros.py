
import numpy as np
from typing import List, Optional
from collections import Counter

def solve_one(inp, color=3):
    h, w = inp.shape
    out = inp.copy()
    
    fill_r = []
    for r in range(h):
        nz = np.argwhere(inp[r] != 0).flatten()
        if len(nz) >= 2:
            s, e = nz[0], nz[-1]
            if np.all(inp[r, s+1:e] == 0):
                fill_r.append((r, s, e))
                
    fill_c = []
    for c in range(w):
        nz = np.argwhere(inp[:, c] != 0).flatten()
        if len(nz) >= 2:
            s, e = nz[0], nz[-1]
            if np.all(inp[s+1:e, c] == 0):
                fill_c.append((c, s, e))
                
    for r, s, e in fill_r:
        out[r, s+1:e] = color
    for c, s, e in fill_c:
        out[s+1:e, c] = color
        
    return out

def solve_fill_empty_gap_between_nonzeros(solver) -> Optional[List[np.ndarray]]:
    """
    Fills any row or column that has a gap of only 0s between two non-zero colors with a specific color.
    """
    if not solver.pairs: return None
    inp_0, out_0 = solver.pairs[0]
    diff = (out_0 != inp_0)
    if not np.any(diff): return None
    
    changed_colors = out_0[diff]
    counts = Counter(changed_colors.flatten())
    if 0 in counts: del counts[0]
    if not counts: return None
    color = int(counts.most_common(1)[0][0])
    
    # Verify
    for inp, out in solver.pairs:
        if not np.array_equal(solve_one(inp, color), out):
            return None
            
    return [solve_one(ti, color) for ti in solver.test_in]
