
import numpy as np

def solve_recolor_to_nearest_full_line(solver):
    def apply_logic(inp):
        inp = np.array(inp)
        h, w = inp.shape
        outp = inp.copy()
        
        full_rows = []
        for r in range(h):
            u = np.unique(inp[r])
            if len(u) == 1 and u[0] != 0: full_rows.append((r, u[0]))
        full_cols = []
        for c in range(w):
            u = np.unique(inp[:, c])
            if len(u) == 1 and u[0] != 0: full_cols.append((c, u[0]))
            
        if not full_rows and not full_cols: return None
        
        # In this task, markers are color 3
        marker_coords = np.argwhere(inp == 3)
        if len(marker_coords) == 0: return None
        
        for r, c in marker_coords:
            best_dist = float('inf')
            best_color = 3
            for fr, fcolor in full_rows:
                dist = abs(r - fr)
                if dist < best_dist: best_dist = dist; best_color = fcolor
            for fc, fcolor in full_cols:
                dist = abs(c - fc)
                if dist < best_dist: best_dist = dist; best_color = fcolor
            outp[r, c] = best_color
        return outp

    results = []
    for inp, outp in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, outp):
            return None
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
