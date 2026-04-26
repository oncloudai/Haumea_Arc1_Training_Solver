import numpy as np
from typing import List, Optional

def solve_tiled_xor_parity(solver) -> Optional[List[np.ndarray]]:
    # For 10fcaaa3
    bg_color = 8
    for inp, out in solver.pairs:
        h, w = inp.shape
        if out.shape[0] % h != 0 or out.shape[1] % w != 0: return None
    
    def process(ti, ho, wo):
        h, w = ti.shape; res = np.tile(ti, (ho // h, wo // w))
        seeds = np.argwhere(ti != 0)
        if len(seeds) == 0: return res
        for r in range(ho):
            for c in range(wo):
                if res[r, c] == 0:
                    val = False
                    for sr, sc in seeds:
                        val ^= ((abs(r - sr) % 2 != 0) ^ (abs(c - sc) % 2 != 0))
                    if not val: res[r, c] = bg_color
        return res

    for inp, out in solver.pairs:
        if not np.array_equal(process(inp, out.shape[0], out.shape[1]), out): return None
        
    return [process(ti, ti.shape[0]*2, ti.shape[1]*2) for ti in solver.test_in]
