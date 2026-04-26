
import numpy as np

def solve_project_pixels_to_block(solver):
    def apply_logic(inp):
        inp = np.array(inp)
        h, w = inp.shape
        
        # Find the main block (most frequent non-zero color that forms a block)
        # In this task it's color 8
        coords8 = np.argwhere(inp == 8)
        if len(coords8) == 0: return None
        r_min, c_min = coords8.min(axis=0)
        r_max, c_max = coords8.max(axis=0)
        
        outp = inp.copy()
        for r in range(h):
            for c in range(w):
                color = inp[r, c]
                if color != 0 and color != 8:
                    tr, tc = r, c
                    if c_min <= c <= c_max:
                        tr = r_min if r < r_min else r_max
                        tc = c
                    elif r_min <= r <= r_max:
                        tc = c_min if c < c_min else c_max
                        tr = r
                    else:
                        # Diagonally away? Not handled yet, but not seen in tasks
                        continue
                    
                    if 0 <= tr < h and 0 <= tc < w:
                        outp[tr, tc] = color
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
