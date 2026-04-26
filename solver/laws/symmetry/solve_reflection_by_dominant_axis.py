import numpy as np
from typing import List, Optional
from collections import Counter

def find_dominant_axis(grid):
    counts = Counter()
    # Exclude 0 (background)
    unq = [c for c in np.unique(grid) if c != 0]
    for color in unq:
        coords = np.argwhere(grid == color)
        if coords.size == 0: continue
        for r in np.unique(coords[:,0]):
            row_cols = sorted(coords[coords[:,0] == r, 1].tolist())
            if len(row_cols) < 2: continue
            mid = (min(row_cols) + max(row_cols)) / 2.0
            is_sym = True
            for c in row_cols:
                if round(2*mid - c) not in row_cols:
                    is_sym = False; break
            if is_sym:
                counts[(color, mid)] += 1
    
    if not counts: return None, None
    (color, axis), freq = counts.most_common(1)[0]
    return color, axis

def solve_reflection_by_dominant_axis(solver) -> Optional[List[np.ndarray]]:
    """
    For each grid, finds the most frequent symmetry axis of some color,
    and reflects all other non-zero pixels across that axis.
    """
    consistent = True
    for inp, out in solver.pairs:
        if inp.shape != out.shape: consistent = False; break
        source_color, axis_c = find_dominant_axis(inp)
        if axis_c is None: consistent = False; break
        
        pred = inp.copy()
        for r in range(inp.shape[0]):
            for c in range(inp.shape[1]):
                val = inp[r, c]
                if val != 0 and val != source_color:
                    c_ref = int(round(2 * axis_c - c))
                    if 0 <= c_ref < inp.shape[1]:
                        pred[r, c] = inp[r, c_ref]
                    else:
                        pred[r, c] = 0
        
        if not np.array_equal(pred, out):
            consistent = False; break
    
    if consistent:
        results = []
        for ti in solver.test_in:
            source_color, axis_c = find_dominant_axis(ti)
            if axis_c is None: return None
            res = ti.copy()
            for r in range(ti.shape[0]):
                for c in range(ti.shape[1]):
                    val = ti[r, c]
                    if val != 0 and val != source_color:
                        c_ref = int(round(2 * axis_c - c))
                        if 0 <= c_ref < ti.shape[1]:
                            res[r, c] = ti[r, c_ref]
                        else:
                            res[r, c] = 0
            results.append(res)
        return results

    return None
