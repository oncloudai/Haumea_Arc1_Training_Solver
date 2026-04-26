
import numpy as np
from typing import List, Optional

def solve_fill_gap_with_new_color(solver) -> Optional[List[np.ndarray]]:
    """
    Find two pixels of the same color in a row/column and fill the zeros between them
    with a specific new color, provided the gap size is within a certain limit.
    """
    for m_color in range(1, 10):
        for fill_color in range(1, 10):
            if m_color == fill_color: continue
            for max_gap in range(1, 10):
                consistent = True; found_any = False
                for pair_idx, (inp, out) in enumerate(solver.pairs):
                    res = inp.copy()
                    h, w = inp.shape
                    for r in range(h):
                        cols = np.where(inp[r, :] == m_color)[0]
                        if len(cols) >= 2:
                            for i in range(len(cols) - 1):
                                c1, c2 = cols[i], cols[i+1]
                                if 0 < c2 - c1 - 1 <= max_gap and np.all(inp[r, c1+1:c2] == 0):
                                    res[r, c1+1:c2] = fill_color
                                    found_any = True
                    for c in range(w):
                        rows = np.where(inp[:, c] == m_color)[0]
                        if len(rows) >= 2:
                            for i in range(len(rows) - 1):
                                r1, r2 = rows[i], rows[i+1]
                                if 0 < r2 - r1 - 1 <= max_gap and np.all(inp[r1+1:r2, c] == 0):
                                    res[r1+1:r2, c] = fill_color
                                    found_any = True
                    if not np.array_equal(res, out):
                        consistent = False; break
                if consistent and found_any:
                    results = []
                    for ti in solver.test_in:
                        res = ti.copy()
                        h, w = ti.shape
                        for r in range(h):
                            cols = np.where(ti[r, :] == m_color)[0]
                            if len(cols) >= 2:
                                for i in range(len(cols) - 1):
                                    c1, c2 = cols[i], cols[i+1]
                                    if 0 < c2 - c1 - 1 <= max_gap and np.all(ti[r, c1+1:c2] == 0):
                                        res[r, c1+1:c2] = fill_color
                        for c in range(w):
                            rows = np.where(ti[:, c] == m_color)[0]
                            if len(rows) >= 2:
                                for i in range(len(rows) - 1):
                                    r1, r2 = rows[i], rows[i+1]
                                    if 0 < r2 - r1 - 1 <= max_gap and np.all(ti[r1+1:r2, c] == 0):
                                        res[r1+1:r2, c] = fill_color
                        results.append(res)
                    return results
    return None
