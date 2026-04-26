import numpy as np
from typing import List, Optional

def solve_grid_sync(solver) -> Optional[List[np.ndarray]]:
    for divider_color in range(1, 10):
        consistent = True; found_any = False
        for inp, out in solver.pairs:
            h, w = inp.shape
            div_rows = [r for r in range(h) if np.all(inp[r, :] == divider_color)]
            div_cols = [c for c in range(w) if np.all(inp[:, c] == divider_color)]
            
            if not div_rows and not div_cols: consistent = False; break
            
            # Find the source segment
            # We can use the segments defined by dividers
            r_bounds = [-1] + div_rows + [h]
            c_bounds = [-1] + div_cols + [w]
            
            segments = []
            for i in range(len(r_bounds) - 1):
                for j in range(len(c_bounds) - 1):
                    r0, r1 = r_bounds[i] + 1, r_bounds[i+1]
                    c0, c1 = c_bounds[j] + 1, c_bounds[j+1]
                    if r1 > r0 and c1 > c0:
                        segments.append(((r0, r1, c0, c1), inp[r0:r1, c0:c1]))
            
            non_empty = [s for s in segments if np.any(s[1] != 0)]
            if not non_empty: consistent = False; break
            
            # Assuming all non-empty segments are consistent if there are multiple
            src_seg = non_empty[0][1]
            
            pred = inp.copy()
            for (r0, r1, c0, c1), seg in segments:
                if seg.shape == src_seg.shape:
                    pred[r0:r1, c0:c1] = src_seg
                    found_any = True
                else:
                    # Inconsistent segment sizes?
                    consistent = False; break
            
            if not consistent or not np.array_equal(pred, out):
                consistent = False; break
                
        if consistent and found_any:
            results = []
            for ti in solver.test_in:
                h, w = ti.shape
                div_rows = [r for r in range(h) if np.all(ti[r, :] == divider_color)]
                div_cols = [c for c in range(w) if np.all(ti[:, c] == divider_color)]
                r_bounds = [-1] + div_rows + [h]
                c_bounds = [-1] + div_cols + [w]
                segments = []
                for i in range(len(r_bounds) - 1):
                    for j in range(len(c_bounds) - 1):
                        r0, r1 = r_bounds[i] + 1, r_bounds[i+1]
                        c0, c1 = c_bounds[j] + 1, c_bounds[j+1]
                        if r1 > r0 and c1 > c0:
                            segments.append(((r0, r1, c0, c1), ti[r0:r1, c0:c1]))
                non_empty = [s for s in segments if np.any(s[1] != 0)]
                if not non_empty: results.append(ti.copy()); continue
                src_seg = non_empty[0][1]
                res = ti.copy()
                for (r0, r1, c0, c1), seg in segments:
                    if seg.shape == src_seg.shape: res[r0:r1, c0:c1] = src_seg
                results.append(res)
            return results
    return None
