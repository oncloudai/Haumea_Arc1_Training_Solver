import numpy as np
from typing import List, Optional
from solver.utils import get_subgrids_by_bg

def solve_subgrid_unique_content(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        consistent = True; results = []
        for inp, out in solver.pairs:
            subs = get_subgrids_by_bg(inp, bg)
            subs = [s for s in subs if s['h'] > 1 and s['w'] > 1]
            if len(subs) < 2: consistent = False; break
            
            unique_subs = []
            for i, s1 in enumerate(subs):
                is_unique = True
                for j, s2 in enumerate(subs):
                    if i == j: continue
                    if s1['h'] == s2['h'] and s1['w'] == s2['w'] and np.array_equal(s1['grid'], s2['grid']):
                        is_unique = False; break
                if is_unique: unique_subs.append(s1)
            
            if len(unique_subs) != 1 or not np.array_equal(unique_subs[0]['grid'], out):
                consistent = False; break
        
        if consistent:
            for ti in solver.test_in:
                subs = get_subgrids_by_bg(ti, bg)
                subs = [s for s in subs if s['h'] > 1 and s['w'] > 1]
                unique_subs = []
                for i, s1 in enumerate(subs):
                    is_unique = True
                    for j, s2 in enumerate(subs):
                        if i == j: continue
                        if s1['h'] == s2['h'] and s1['w'] == s2['w'] and np.array_equal(s1['grid'], s2['grid']):
                            is_unique = False; break
                    if is_unique: unique_subs.append(s1)
                if len(unique_subs) == 1: results.append(unique_subs[0]['grid'])
                else: break
            if len(results) == len(solver.test_in): return results
    return None
