import numpy as np
from scipy.ndimage import label
from typing import List, Optional

def get_objs_8a004b2b(grid, colors):
    objs = {}
    for c in colors:
        mask = (grid == c)
        labeled, num = label(mask)
        for i in range(1, num + 1):
            coords = np.argwhere(labeled == i)
            r_min, c_min = coords.min(axis=0); r_max, c_max = coords.max(axis=0)
            offsets = frozenset((int(r - r_min), int(c - c_min)) for r, c in coords)
            if c not in objs: objs[c] = []
            objs[c].append({'r_min': r_min, 'c_min': c_min, 'r_max': r_max, 'c_max': c_max, 'offsets': offsets, 'coords': coords, 'color': int(c)})
    return objs

def solve_grid_8a004b2b(grid):
    grid = np.array(grid)
    yellow = np.argwhere(grid == 4)
    if yellow.size == 0: return grid
    ry_min, cy_min = yellow.min(axis=0); ry_max, cy_max = yellow.max(axis=0)
    out_h, out_w = ry_max - ry_min + 1, cy_max - cy_min + 1
    unique_colors = np.unique(grid)
    unique_colors = unique_colors[(unique_colors != 0) & (unique_colors != 4)]
    all_objs = get_objs_8a004b2b(grid, unique_colors)
    stamps, skeleton_pixels = [], []
    for c, c_objs in all_objs.items():
        for obj in c_objs:
            if ry_min <= obj['r_min'] <= ry_max and cy_min <= obj['c_min'] <= cy_max: stamps.append(obj)
            elif obj['r_min'] > ry_max:
                for r, c_pos in obj['coords']: skeleton_pixels.append((int(r), int(c_pos), int(c)))
    if not skeleton_pixels or not stamps: return grid[ry_min:ry_max+1, cy_min:cy_max+1]
    sh, sw = stamps[0]['r_max'] - stamps[0]['r_min'] + 1, stamps[0]['c_max'] - stamps[0]['c_min'] + 1
    votes = {}
    for s in stamps:
        ri, ci = s['r_min'] - ry_min, s['c_min'] - cy_min
        for rs, cs, col in skeleton_pixels:
            if col == s['color']:
                p_or, p_oc = ri - rs * sh, ci - cs * sw
                votes[(p_or, p_oc)] = votes.get((p_or, p_oc), 0) + 1
    best_or, best_oc = max(votes.items(), key=lambda x: x[1])[0]
    out = np.zeros((out_h, out_w), dtype=int)
    for r, c in yellow: out[r - ry_min, c - cy_min] = 4
    color_to_offsets = {s['color']: s['offsets'] for s in stamps}
    for rs, cs, color in skeleton_pixels:
        tr, tc = rs * sh + best_or, cs * sw + best_oc
        offsets = color_to_offsets.get(color, frozenset((i, j) for i in range(sh) for j in range(sw)))
        for dr, dc in offsets:
            nr, nc = tr + dr, tc + dc
            if 0 <= nr < out_h and 0 <= nc < out_w:
                if out[nr, nc] != 4: out[nr, nc] = color
    return out

def solve_skeleton_stamp_composition(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_8a004b2b(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_8a004b2b(ti) for ti in solver.test_in]
    return None
