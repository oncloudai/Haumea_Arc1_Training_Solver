
import numpy as np

def solve_cross_with_intersection_recolor(solver):
    def apply_logic(inp):
        inp = np.array(inp)
        h, w = inp.shape
        outp = np.zeros_like(inp)
        pixels = np.argwhere(inp != 0)
        for r, c in pixels:
            color = inp[r, c]
            for i in range(h):
                if outp[i, c] == 0: outp[i, c] = color
                elif outp[i, c] != color: outp[i, c] = 2
            for j in range(w):
                if outp[r, j] == 0: outp[r, j] = color
                elif outp[r, j] != color: outp[r, j] = 2
        return outp

    results = []
    for inp, outp in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, outp): return None
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
