
import numpy as np

def solve_extract_smallest_color_bbox(solver):
    def apply_logic(inp):
        inp = np.array(inp)
        unique, counts = np.unique(inp, return_counts=True)
        counts = counts[unique != 0]
        unique = unique[unique != 0]
        if len(unique) == 0: return None
        min_idx = np.argmin(counts)
        target_color = unique[min_idx]
        coords = np.argwhere(inp == target_color)
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        return inp[r_min:r_max+1, c_min:c_max+1]

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
