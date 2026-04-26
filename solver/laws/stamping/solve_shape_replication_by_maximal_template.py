import numpy as np
from scipy.ndimage import label

def get_objects(grid):
    labeled, num = label(grid != 0)
    objs = []
    for i in range(1, num + 1):
        coords = np.argwhere(labeled == i)
        r_min, r_max = coords[:, 0].min(), coords[:, 0].max()
        c_min, c_max = coords[:, 1].min(), coords[:, 1].max()
        obj_grid = grid[r_min:r_max+1, c_min:c_max+1].copy()
        obj_mask = labeled[r_min:r_max+1, c_min:c_max+1] == i
        obj_grid[~obj_mask] = 0
        objs.append({'grid': obj_grid, 'r': r_min, 'c': c_min, 'h': r_max-r_min+1, 'w': c_max-c_min+1})
    return objs

def solve_shape_replication_by_maximal_template_logic(input_grid):
    objs = get_objects(input_grid)
    if len(objs) < 2: return input_grid
    
    counts = {}
    for obj in objs:
        for c in np.unique(obj['grid']):
            if c != 0:
                counts[c] = counts.get(c, 0) + 1
    
    if not counts: return input_grid
    marker_color = -1
    max_objs = 0
    for c, count in counts.items():
        if count > max_objs:
            max_objs = count
            marker_color = c
            
    group = [o for o in objs if marker_color in o['grid']]
    if len(group) < 2: return input_grid
    
    structures = []
    for o in group:
        m_coords = np.argwhere(o['grid'] == marker_color)
        m_r, m_c = m_coords[:, 0].min(), m_coords[:, 1].min()
        m_h = m_coords[:, 0].max() - m_r + 1
        m_w = m_coords[:, 1].max() - m_c + 1
        
        other_pixels = np.argwhere((o['grid'] != 0) & (o['grid'] != marker_color))
        slots = set()
        p_colors = []
        if len(other_pixels) > 0:
            p_colors = np.unique(o['grid'][other_pixels[:, 0], other_pixels[:, 1]])
            for pr, pc in other_pixels:
                slots.add(( (pr - m_r) // m_h, (pc - m_c) // m_w ))
        
        structures.append({
            'obj': o,
            'marker_r': m_r, 'marker_c': m_c,
            'marker_h': m_h, 'marker_w': m_w,
            'slots': slots,
            'p_colors': p_colors
        })
        
    template = max(structures, key=lambda x: len(x['slots']))
    output_grid = input_grid.copy()
    
    for target in structures:
        if len(target['p_colors']) == 0: continue
        p_color = target['p_colors'][0]
        anchor_r = target['obj']['r'] + target['marker_r']
        anchor_c = target['obj']['c'] + target['marker_c']
        th, tw = target['marker_h'], target['marker_w']
        for dr, dc in template['slots']:
            nr, nc = anchor_r + dr * th, anchor_c + dc * tw
            for ir in range(th):
                for ic in range(tw):
                    if 0 <= nr + ir < output_grid.shape[0] and 0 <= nc + ic < output_grid.shape[1]:
                        output_grid[nr + ir, nc + ic] = p_color
                        
    return output_grid

def solve_shape_replication_by_maximal_template(solver):
    preds = []
    for input_grid, output_grid in solver.pairs:
        p = solve_shape_replication_by_maximal_template_logic(input_grid)
        if not np.array_equal(p, output_grid):
            return None
    for input_grid in solver.test_in:
        preds.append(solve_shape_replication_by_maximal_template_logic(input_grid))
    return preds
