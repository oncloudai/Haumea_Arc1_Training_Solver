
import numpy as np
from scipy.ndimage import label

def solve_blob_perpendicular_reflection(solver):
    def apply_logic(inp):
        inp = np.array(inp)
        h, w = inp.shape
        outp = inp.copy()
        mask = (inp == 3)
        labeled, n = label(mask, structure=np.ones((3,3)))
        if n == 0: return None
        for lbl in range(1, n+1):
            coords = np.argwhere(labeled == lbl)
            if len(coords) < 2: continue
            max_dist = -1
            p1, p2 = None, None
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    d = np.sum((coords[i] - coords[j])**2)
                    if d > max_dist: max_dist = d; p1, p2 = coords[i], coords[j]
            b1_coords, b2_coords = [], []
            for c in coords:
                if np.sum((c - p1)**2) <= np.sum((c - p2)**2): b1_coords.append(c)
                else: b2_coords.append(c)
            b1_coords, b2_coords = np.array(b1_coords), np.array(b2_coords)
            c1, c2 = np.mean(b1_coords, axis=0), np.mean(b2_coords, axis=0)
            center = (c1 + c2) / 2.0
            v = c2 - c1
            v_perp = np.array([v[1], -v[0]])
            new_centers = [center + 1.5 * v_perp, center - 1.5 * v_perp]
            rel_coords = b1_coords - c1
            for nc in new_centers:
                for dr, dc in rel_coords:
                    tr, tc = int(round(nc[0] + dr)), int(round(nc[1] + dc))
                    if 0 <= tr < h and 0 <= tc < w: outp[tr, tc] = 8
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
