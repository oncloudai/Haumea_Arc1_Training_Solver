
import numpy as np

def solve_extract_quadrant(solver):
    def apply_logic(inp):
        inp = np.array(inp)
        coords = np.argwhere(inp != 0)
        if len(coords) == 0: return None
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        oh = r_max - r_min + 1
        ow = c_max - c_min + 1
        qh, qw = oh // 2, ow // 2
        if qh == 0 or qw == 0: return None
        return inp[r_min:r_min+qh, c_min:c_min+qw]

    results = []
    for inp, outp in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != outp.shape or not np.array_equal(pred, outp):
            return None
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
