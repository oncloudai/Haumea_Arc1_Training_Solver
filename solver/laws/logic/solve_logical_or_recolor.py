import numpy as np
from typing import List, Optional

def solve_logical_or_recolor(solver) -> Optional[List[np.ndarray]]:
    for separator_color in range(1, 10):
        consistent = True; found_any = False; fill_color = 3
        for inp, out in solver.pairs:
            h, w = inp.shape
            sep_rows = [r for r in range(h) if np.all(inp[r, :] == separator_color)]
            if len(sep_rows) != 1: consistent = False; break
            
            sr = sep_rows[0]
            p1 = inp[0:sr, :]
            p2 = inp[sr+1:2*sr+1, :] if 2*sr+1 <= h else inp[sr+1:, :]
            
            if p1.shape != out.shape or p2.shape != out.shape:
                consistent = False; break
            
            pred = np.where((p1 != 0) | (p2 != 0), fill_color, 0)
            if not np.array_equal(pred, out):
                consistent = False; break
            found_any = True
            
        if consistent and found_any:
            results = []
            for ti in solver.test_in:
                h, w = ti.shape
                sep_rows = [r for r in range(h) if np.all(ti[r, :] == separator_color)]
                if len(sep_rows) != 1: results.append(ti.copy()); continue
                sr = sep_rows[0]; p1 = ti[0:sr, :]; p2 = ti[sr+1:2*sr+1, :] if 2*sr+1 <= h else ti[sr+1:, :]
                res = np.where((p1 != 0) | (p2 != 0), fill_color, 0)
                results.append(res)
            return results
    return None
