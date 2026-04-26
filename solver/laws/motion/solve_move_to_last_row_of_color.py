import numpy as np
from typing import List, Optional

def solve_move_to_last_row_of_color(solver) -> Optional[List[np.ndarray]]:
    """
    Finds a source color (e.g. 1) and a target color (e.g. 5).
    For each source pixel, moves it to the bottom-most row that contains the target color
    in the same column.
    """
    # 1. Infer source, target and background
    if not solver.pairs: return None
    inp0, out0 = solver.pairs[0]
    unq_in = np.unique(inp0)
    unq_out = np.unique(out0)
    
    # Source color is present in In but moves position in Out
    # Or just disappears from its original place.
    # Let's try all pairs of colors as (source, target)
    for source in unq_in:
        if source == 0: continue
        for target in unq_in:
            if target == source or target == 0: continue
            
            consistent = True
            found_any = False
            for inp, out in solver.pairs:
                if inp.shape != out.shape: consistent = False; break
                
                pred = inp.copy()
                # Find source pixels
                src_coords = np.argwhere(inp == source)
                if len(src_coords) == 0:
                    if not np.array_equal(inp, out): consistent = False; break
                    continue
                
                # For each source pixel, find bottom-most target in same col
                for sr, sc in src_coords:
                    target_rows = np.where(inp[:, sc] == target)[0]
                    if len(target_rows) == 0:
                        # If no target in col, maybe it stays or disappears?
                        # In this task, there's always a target.
                        consistent = False; break
                    
                    last_r = target_rows.max()
                    pred[sr, sc] = inp[0, 0] # Assume background is at (0,0) or inferred
                    # Wait, background might not be 0. Let's use 0 for now.
                    pred[sr, sc] = 0
                    pred[last_r, sc] = source
                    found_any = True
                
                if not np.array_equal(pred, out):
                    consistent = False; break
            
            if consistent and found_any:
                results = []
                for ti in solver.test_in:
                    res = ti.copy()
                    src_coords = np.argwhere(ti == source)
                    for sr, sc in src_coords:
                        target_rows = np.where(ti[:, sc] == target)[0]
                        if len(target_rows) > 0:
                            last_r = target_rows.max()
                            res[sr, sc] = 0
                            res[last_r, sc] = source
                    results.append(res)
                return results
    return None
