import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_project_colors_to_borders(solver) -> Optional[List[np.ndarray]]:
    """
    Learns how to project non-central objects to the borders of a new grid.
    Tests different border assignments and modulo offsets.
    """
    def get_bbox(mask):
        rows, cols = np.where(mask)
        if len(rows) == 0: return None
        return rows.min(), rows.max(), cols.min(), cols.max()

    def apply_logic(grid, params):
        grid = np.array(grid)
        h, w = grid.shape
        bbox8 = get_bbox(grid == 8)
        if not bbox8: return None
        r1_8, r2_8, c1_8, c2_8 = bbox8
        S_r = r2_8 - r1_8 + 1
        S_c = c2_8 - c1_8 + 1
        out_h, out_w = S_r + 2, S_c + 2
        
        out = np.zeros((out_h, out_w), dtype=int)
        for r in range(r1_8, r2_8 + 1):
            for c in range(c1_8, c2_8 + 1):
                if grid[r, c] == 8:
                    out[r - r1_8 + 1, c - c1_8 + 1] = 8
                    
        other_colors = [c for c in np.unique(grid) if c != 0 and c != 8]
        horiz_info = []
        vert_info = []
        for color in other_colors:
            bbox = get_bbox(grid == color)
            if (bbox[3] - bbox[2]) >= (bbox[1] - bbox[0]):
                horiz_info.append({'color': color, 'bbox': bbox, 'min_r': bbox[0]})
            else:
                vert_info.append({'color': color, 'bbox': bbox, 'min_c': bbox[2]})
        
        horiz_info.sort(key=lambda x: x['min_r'])
        vert_info.sort(key=lambda x: x['min_c'])
        
        if len(horiz_info) == 2:
            for i, info in enumerate(horiz_info):
                target_r = 0 if i == params['h_perm'][0] else out_h - 1
                for c in range(info['bbox'][2], info['bbox'][3] + 1):
                    if np.any(grid[info['bbox'][0]:info['bbox'][1]+1, c] == info['color']):
                        c_out = (c - c1_8 + params['h_off']) % out_w
                        out[target_r, c_out] = info['color']
        
        if len(vert_info) == 2:
            for i, info in enumerate(vert_info):
                target_c = 0 if i == params['v_perm'][0] else out_w - 1
                for r in range(info['bbox'][0], info['bbox'][1] + 1):
                    if np.any(grid[r, info['bbox'][2]:info['bbox'][3]+1] == info['color']):
                        r_out = (r - r1_8 + params['v_off']) % out_h
                        out[r_out, target_c] = info['color']
        return out

    # Search space for params
    for hp0 in [0, 1]:
        for vp0 in [0, 1]:
            for ho in range(10):
                for vo in range(10):
                    p = {'h_perm': [hp0, 1-hp0], 'v_perm': [vp0, 1-vp0], 'h_off': ho, 'v_off': vo}
                    consistent = True
                    for inp, out_expected in solver.pairs:
                        pred = apply_logic(inp, p)
                        if pred is None or not np.array_equal(pred, out_expected):
                            consistent = False; break
                    if consistent:
                        return [apply_logic(ti, p) for ti in solver.test_in]
    return None
