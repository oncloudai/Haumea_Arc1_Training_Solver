import numpy as np
from typing import List, Optional

def solve_grid_fill_between_points(solver) -> Optional[List[np.ndarray]]:
    for m_color in range(10):
        for fill_color in range(10):
            consistent = True; found_any = False
            for inp, out in solver.pairs:
                res = inp.copy(); coords = np.argwhere(inp == m_color)
                if len(coords) < 2: consistent = False; break
                for i in range(len(coords)):
                    for j in range(i+1, len(coords)):
                        r1, c1, r2, c2 = coords[i][0], coords[i][1], coords[j][0], coords[j][1]
                        if r1 == r2: res[r1, min(c1,c2)+1:max(c1,c2)] = fill_color; found_any = True
                        elif c1 == c2: res[min(r1,r2)+1:max(r1,r2), c1] = fill_color; found_any = True
                if not np.array_equal(res, out): consistent = False; break
            if consistent and found_any:
                results = []
                for ti in solver.test_in:
                    res = ti.copy(); coords = np.argwhere(ti == m_color)
                    for i in range(len(coords)):
                        for j in range(i+1, len(coords)):
                            r1, c1, r2, c2 = coords[i][0], coords[i][1], coords[j][0], coords[j][1]
                            if r1 == r2: res[r1, min(c1,c2)+1:max(c1,c2)] = fill_color
                            elif c1 == c2: res[min(r1,r2)+1:max(r1,r2), c1] = fill_color
                    results.append(res)
                return results
    return None
