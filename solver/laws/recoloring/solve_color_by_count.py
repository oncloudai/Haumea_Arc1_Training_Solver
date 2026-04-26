import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_color_by_count(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        for conn in [4, 8]:
            consistent = True; mapping = {}
            for inp, out in solver.pairs:
                count = len(get_blobs(inp, bg, conn))
                if out.size != 1: consistent = False; break
                c = int(out.flatten()[0])
                if count in mapping and mapping[count] != c: consistent = False; break
                mapping[count] = c
            if consistent and mapping:
                results = []
                for ti in solver.test_in:
                    cnt = len(get_blobs(ti, bg, conn))
                    res = np.full(solver.train_out[0].shape, mapping.get(cnt, 0))
                    results.append(res)
                return results
    return None
