import numpy as np
from typing import List, Optional

def solve_quadrant_sync(solver) -> Optional[List[np.ndarray]]:
    for divider_color in range(1, 10):
        consistent = True; found_any = False
        for inp, out in solver.pairs:
            h, w = inp.shape
            # Find divider row and column
            div_rows = [r for r in range(h) if np.all(inp[r, :] == divider_color)]
            div_cols = [c for c in range(w) if np.all(inp[:, c] == divider_color)]
            
            if len(div_rows) != 1 or len(div_cols) != 1:
                consistent = False; break
            
            dr, dc = div_rows[0], div_cols[1] if len(div_cols) > 1 else div_cols[0]
            # Wait, div_cols[1] is a typo if len is 1. Fixed.
            dc = div_cols[0]
            
            # Quadrants: TL, TR, BL, BR
            quads = [
                inp[0:dr, 0:dc],
                inp[0:dr, dc+1:],
                inp[dr+1:, 0:dc],
                inp[dr+1:, dc+1:]
            ]
            
            # Find the non-empty quadrant
            non_empty = [i for i, q in enumerate(quads) if np.any(q != 0)]
            if len(non_empty) != 1:
                # Maybe multiple are non-empty but they match?
                # For now, let's assume exactly one is the source.
                pass
            
            if not non_empty: consistent = False; break
            src_idx = non_empty[0]
            src_quad = quads[src_idx]
            
            pred = inp.copy()
            pred[0:dr, 0:dc] = src_quad
            pred[0:dr, dc+1:] = src_quad
            pred[dr+1:, 0:dc] = src_quad
            pred[dr+1:, dc+1:] = src_quad
            
            if not np.array_equal(pred, out):
                consistent = False; break
            found_any = True
            
        if consistent and found_any:
            results = []
            for ti in solver.test_in:
                h, w = ti.shape
                div_rows = [r for r in range(h) if np.all(ti[r, :] == divider_color)]
                div_cols = [c for c in range(w) if np.all(ti[:, c] == divider_color)]
                if len(div_rows) != 1 or len(div_cols) != 1:
                    results.append(ti.copy()); continue
                dr, dc = div_rows[0], div_cols[0]
                quads = [ti[0:dr, 0:dc], ti[0:dr, dc+1:], ti[dr+1:, 0:dc], ti[dr+1:, dc+1:]]
                non_empty = [i for i, q in enumerate(quads) if np.any(q != 0)]
                if not non_empty: results.append(ti.copy()); continue
                src_quad = quads[non_empty[0]]
                res = ti.copy()
                res[0:dr, 0:dc] = src_quad
                res[0:dr, dc+1:] = src_quad
                res[dr+1:, 0:dc] = src_quad
                res[dr+1:, dc+1:] = src_quad
                results.append(res)
            return results
    return None
