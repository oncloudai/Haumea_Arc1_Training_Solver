import numpy as np
from typing import List, Optional

def solve_region_frame_expansion(solver) -> Optional[List[np.ndarray]]:
    for inp, out in solver.pairs:
        h, w = inp.shape; coords = np.argwhere(inp != 0)
        if len(coords) < 2: return None
        if h != 10 or len(coords) != 2: return None
    results = []
    for ti in solver.test_in:
        h, w = ti.shape; res = np.zeros_like(ti); seeds = sorted([{'r':r,'c':c,'color':ti[r,c]} for r,c in np.argwhere(ti!=0)], key=lambda x:x['r']); rh = h // len(seeds)
        for i, s in enumerate(seeds):
            r1, r2 = i*rh, (i+1)*rh-1; res[s['r'], :] = s['color']; res[r1 if i==0 else r2, :] = s['color']; res[r1:r2+1, 0] = s['color']; res[r1:r2+1, w-1] = s['color']
        results.append(res)
    return results
