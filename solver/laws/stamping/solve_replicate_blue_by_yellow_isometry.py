import numpy as np
from scipy.ndimage import label
import itertools
from typing import List, Optional

def get_objects_7df24a62(grid, color):
    mask = (grid == color)
    labeled, num = label(mask)
    objs = []
    for i in range(1, num + 1):
        coords = np.argwhere(labeled == i)
        r_min, c_min = coords.min(axis=0)
        objs.append({'coords': coords, 'r_min': r_min, 'c_min': c_min})
    return objs

def get_isometry_7df24a62(pts1, pts2):
    transforms = [lambda r, c: (r, c), lambda r, c: (c, -r), lambda r, c: (-r, -c), lambda r, c: (-c, r),
                  lambda r, c: (r, -c), lambda r, c: (-r, c), lambda r, c: (c, r), lambda r, c: (-c, -r)]
    pts2_sorted = sorted(list(pts2))
    for trans in transforms:
        for p_pts1 in itertools.permutations(pts1):
            t_pts1 = [trans(r, c) for r, c in p_pts1]
            dr, dc = pts2_sorted[0][0] - t_pts1[0][0], pts2_sorted[0][1] - t_pts1[0][1]
            applied = sorted([(r + dr, c + dc) for r, c in t_pts1])
            if applied == pts2_sorted: return trans, dr, dc
    return None

def solve_grid_7df24a62(grid):
    grid = np.array(grid)
    h, w = grid.shape
    blue_objs = get_objects_7df24a62(grid, 1)
    if not blue_objs: return grid
    src = blue_objs[0]
    all_yellow = set(tuple(p) for p in np.argwhere(grid == 4))
    src_landmarks = [p for p in all_yellow if src['r_min'] <= p[0] <= np.max(src['coords'][:,0]) and src['c_min'] <= p[1] <= np.max(src['coords'][:,1])]
    if not src_landmarks: return grid
    num_l = len(src_landmarks)
    out = grid.copy()
    for target_subset in itertools.combinations(all_yellow, num_l):
        if set(target_subset) == set(src_landmarks): continue
        res = get_isometry_7df24a62(src_landmarks, target_subset)
        if res:
            trans, dr, dc = res
            for r, c in src['coords']:
                tr, tc = trans(r, c)
                nr, nc = tr + dr, tc + dc
                if 0 <= nr < h and 0 <= nc < w: out[nr, nc] = 1
    return out

def solve_replicate_blue_by_yellow_isometry(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_7df24a62(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_7df24a62(ti) for ti in solver.test_in]
    return None
