import numpy as np
from typing import List, Optional
from solver.utils import get_subgrids_by_bg

def solve_largest_subgrid_extraction(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        consistent = True; results = []
        for inp, out in solver.pairs:
            subs = get_subgrids_by_bg(inp, bg)
            if not subs: consistent = False; break
            # Sort by area descending
            subs = sorted(subs, key=lambda s: s['h'] * s['w'], reverse=True)
            if not np.array_equal(subs[0]['grid'], out): consistent = False; break
        if consistent:
            for ti in solver.test_in:
                subs = get_subgrids_by_bg(ti, bg)
                if not subs: break
                subs = sorted(subs, key=lambda s: s['h'] * s['w'], reverse=True)
                results.append(subs[0]['grid'])
            if len(results) == len(solver.test_in): return results
    return None
