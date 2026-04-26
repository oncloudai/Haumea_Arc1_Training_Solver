import numpy as np
from typing import List, Optional

def solve_row_wise_recolor_from_source(solver) -> Optional[List[np.ndarray]]:
    consistent = True; found_any = False
    target_color = 5
    
    for inp, out in solver.pairs:
        h, w = inp.shape
        pred = inp.copy()
        for r in range(h):
            row = inp[r, :]
            # Find source color (non-zero, non-5)
            source_colors = [c for c in np.unique(row) if c != 0 and c != target_color]
            if len(source_colors) == 1:
                src = source_colors[0]
                # Replace target_color with src
                pred[r, row == target_color] = src
                found_any = True
            elif len(source_colors) > 1:
                # Ambiguous source color?
                consistent = False; break
        
        if not np.array_equal(pred, out):
            consistent = False; break
            
    if consistent and found_any:
        results = []
        for ti in solver.test_in:
            h, w = ti.shape
            res = ti.copy()
            for r in range(h):
                row = ti[r, :]
                source_colors = [c for c in np.unique(row) if c != 0 and c != target_color]
                if len(source_colors) == 1:
                    src = source_colors[0]
                    res[r, row == target_color] = src
            results.append(res)
        return results
    return None
