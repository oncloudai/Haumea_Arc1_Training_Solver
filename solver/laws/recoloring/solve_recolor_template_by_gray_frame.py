import numpy as np
from scipy.ndimage import label
from typing import List, Optional

def get_objects_776ffc46(grid):
    grid = np.array(grid)
    mask = (grid != 5) & (grid != 0)
    labeled, num_features = label(mask)
    objs = []
    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled == i)
        r_min, c_min = coords.min(axis=0)
        color = int(grid[coords[0, 0], coords[0, 1]])
        offsets = frozenset((int(r - r_min), int(c - c_min)) for r, c in coords)
        objs.append({'coords': coords, 'color': color, 'offsets': offsets, 'id': i})
    return objs

def is_complete_box_776ffc46(coords, h, w):
    coords = np.array(list(coords))
    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    if r_max - r_min < 2 or c_max - c_min < 2: return False
    perimeter = []
    for r in range(r_min, r_max + 1):
        perimeter.append((r, c_min)); perimeter.append((r, c_max))
    for c in range(c_min + 1, c_max):
        perimeter.append((r_min, c)); perimeter.append((r_max, c))
    found_count = 0
    coords_set = set(tuple(p) for p in coords)
    for p in perimeter:
        if p in coords_set: found_count += 1
    return found_count >= 0.9 * len(perimeter)

def solve_grid_776ffc46(grid):
    grid = np.array(grid)
    h, w = grid.shape
    out = grid.copy()
    gray_mask = (grid == 5)
    labeled_gray, num_gray = label(gray_mask)
    objs = get_objects_776ffc46(grid)
    goal_templates = []
    for i in range(1, num_gray + 1):
        coords_gray = np.argwhere(labeled_gray == i)
        if is_complete_box_776ffc46(coords_gray, h, w):
            r_min, c_min = coords_gray.min(axis=0)
            r_max, c_max = coords_gray.max(axis=0)
            for obj in objs:
                is_inside = True
                for r, c in obj['coords']:
                    if not (r_min < r < r_max and c_min < c < c_max):
                        is_inside = False; break
                if is_inside:
                    goal_templates.append((obj['offsets'], obj['color']))
    for obj in objs:
        for t_offsets, t_color in goal_templates:
            if obj['offsets'] == t_offsets:
                for r, c in obj['coords']:
                    out[r, c] = t_color
                break
    return out

def solve_recolor_template_by_gray_frame(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_776ffc46(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_776ffc46(ti) for ti in solver.test_in]
    return None
