
import numpy as np

def solve_copy_row_by_marker_column(solver):
    def apply_logic(inp):
        inp = np.array(inp)
        h, w = inp.shape
        outp = inp.copy()
        source_row = inp[0, :]
        source_col = inp[:, -1]
        if np.all(source_row == 0) or np.all(source_col == 0): return None
        for r in range(1, h):
            if source_col[r] != 0:
                for c in range(w - 1):
                    if source_row[c] != 0: outp[r, c] = 2
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
