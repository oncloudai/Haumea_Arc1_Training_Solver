
import numpy as np

def solve_ray_from_marker_on_bbox_edge(solver):
    def apply_logic(inp):
        inp = np.array(inp)
        h, w = inp.shape
        unique, counts = np.unique(inp, return_counts=True)
        
        marker_color = None
        for u, c in zip(unique, counts):
            if u != 0 and c == 1:
                marker_color = u; break
        if marker_color is None: return None
        
        marker_pos = np.argwhere(inp == marker_color)[0]
        other_colors = [u for u in unique if u != 0 and u != marker_color]
        if not other_colors: return None
        other_color = other_colors[0]
        other_coords = np.argwhere(inp == other_color)
        r_min, c_min = other_coords.min(axis=0)
        r_max, c_max = other_coords.max(axis=0)
        
        mr, mc = marker_pos
        dr, dc = 0, 0
        # Determine direction: away from the edge the marker is on
        if mr == r_min: dr = 1
        elif mr == r_max: dr = -1
        elif mc == c_min: dc = 1
        elif mc == c_max: dc = -1
        else: return None # Not on an edge
        
        outp = inp.copy()
        curr_r, curr_c = mr, mc
        while True:
            curr_r += dr
            curr_c += dc
            if 0 <= curr_r < h and 0 <= curr_c < w:
                if outp[curr_r, curr_c] == 0:
                    outp[curr_r, curr_c] = marker_color
            else:
                break
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
