import numpy as np
from typing import List, Optional

def solve_marker_neighborhood_union(solver) -> Optional[List[np.ndarray]]:
    bg = 0
    for marker_c in range(1, 10):
        consistent = True; results = []
        for inp, out in solver.pairs:
            m_coords = np.argwhere(inp == marker_c)
            if len(m_coords) == 0: consistent = False; break
            # Create union of 3x3 patches
            res = np.zeros((3, 3), dtype=int)
            for r, c in m_coords:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1]:
                            val = inp[nr, nc]
                            if val != bg:
                                if res[dr+1, dc+1] == 0 or val == marker_c:
                                    res[dr+1, dc+1] = val
            if not np.array_equal(res, out): consistent = False; break
        if consistent:
            for ti in solver.test_in:
                m_coords = np.argwhere(ti == marker_c)
                res = np.zeros((3, 3), dtype=int)
                for r, c in m_coords:
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < ti.shape[0] and 0 <= nc < ti.shape[1]:
                                val = ti[nr, nc]
                                if val != bg:
                                    if res[dr+1, dc+1] == 0 or val == marker_c:
                                        res[dr+1, dc+1] = val
                results.append(res)
            return results
    return None
