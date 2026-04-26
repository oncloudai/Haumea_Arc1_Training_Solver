import numpy as np
from typing import List, Optional

def solve_recolor_diagonal_alternating(solver) -> Optional[List[np.ndarray]]:
    # In this task, the replacement color is always 4.
    # But let's try to be more general if possible.
    # For now, 4 is a safe bet since it's used in all pairs.
    replace_color = 4
    
    consistent = True; found_any = False
    for inp, out in solver.pairs:
        # Find the main color (the one that gets replaced)
        # It's the only non-zero color in the input that is also in the output.
        non_zero_colors = [c for c in np.unique(inp) if c != 0]
        if len(non_zero_colors) != 1: consistent = False; break
        main_color = non_zero_colors[0]
        
        pred = inp.copy()
        coords = np.argwhere(inp == main_color)
        
        # Group by diagonal c - r
        diagonals = {}
        for r, c in coords:
            d = c - r
            if d not in diagonals: diagonals[d] = []
            diagonals[d].append((r, c))
            
        for d in diagonals:
            # Sort by row
            diag_coords = sorted(diagonals[d], key=lambda x: x[0])
            for i in range(1, len(diag_coords), 2):
                r, c = diag_coords[i]
                pred[r, c] = replace_color
        
        if not np.array_equal(pred, out):
            consistent = False; break
        found_any = True
        
    if consistent and found_any:
        results = []
        for ti in solver.test_in:
            non_zero_colors = [c for c in np.unique(ti) if c != 0]
            if len(non_zero_colors) != 1:
                results.append(ti.copy()); continue
            main_color = non_zero_colors[0]
            res = ti.copy()
            coords = np.argwhere(ti == main_color)
            diagonals = {}
            for r, c in coords:
                d = c - r
                if d not in diagonals: diagonals[d] = []
                diagonals[d].append((r, c))
            for d in diagonals:
                diag_coords = sorted(diagonals[d], key=lambda x: x[0])
                for i in range(1, len(diag_coords), 2):
                    r, c = diag_coords[i]
                    res[r, c] = replace_color
            results.append(res)
        return results
    return None
