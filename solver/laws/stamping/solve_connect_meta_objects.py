import numpy as np
from typing import List, Optional

def solve_one(inp):
    inp = np.array(inp)
    h, w = inp.shape
    
    # 1. Identify non-empty columns
    non_empty_cols = []
    for c in range(w):
        if not np.all(inp[:, c] == 0):
            non_empty_cols.append(c)
    
    if not non_empty_cols:
        return inp
    
    # 2. Group adjacent non-empty columns into Meta-Objects
    groups = []
    if non_empty_cols:
        curr_group = [non_empty_cols[0]]
        for i in range(1, len(non_empty_cols)):
            if non_empty_cols[i] == non_empty_cols[i-1] + 1:
                curr_group.append(non_empty_cols[i])
            else:
                groups.append(curr_group)
                curr_group = [non_empty_cols[i]]
        groups.append(curr_group)
    
    # 3. Process each Meta-Object
    meta_objects = []
    for group in groups:
        cols = inp[:, group]
        # Find seed color (most frequent non-zero, non-5)
        nz5_mask = (cols != 0) & (cols != 5)
        if not np.any(nz5_mask):
            seed_color = 5 # Should not happen based on problem
        else:
            colors, counts = np.unique(cols[nz5_mask], return_counts=True)
            seed_color = colors[np.argmax(counts)]
        
        # Identify anchors BEFORE recoloring
        # Left anchor: first 5 in leftmost column. If none, first non-zero in leftmost.
        left_5s = np.argwhere(cols[:, 0] == 5)
        if len(left_5s) > 0:
            left_anchor = left_5s[0, 0]
        else:
            left_nz = np.argwhere(cols[:, 0] != 0)
            left_anchor = left_nz[0, 0] if len(left_nz) > 0 else 0
            
        # Right anchor: first 5 in rightmost column. If none, first non-zero in rightmost.
        right_5s = np.argwhere(cols[:, -1] == 5)
        if len(right_5s) > 0:
            right_anchor = right_5s[0, 0]
        else:
            right_nz = np.argwhere(cols[:, -1] != 0)
            right_anchor = right_nz[0, 0] if len(right_nz) > 0 else 0
            
        # Recolor 5s
        processed_cols = cols.copy()
        processed_cols[processed_cols == 5] = seed_color
            
        meta_objects.append({
            'cols': processed_cols,
            'left_anchor': left_anchor,
            'right_anchor': right_anchor,
            'width': len(group)
        })
        
    # 4. Calculate shifts
    shifts = [0] * len(meta_objects)
    curr_right_anchor = meta_objects[0]['right_anchor']
    for i in range(1, len(meta_objects)):
        # Shift so this object's left anchor matches previous object's right anchor
        shifts[i] = curr_right_anchor - meta_objects[i]['left_anchor']
        curr_right_anchor = meta_objects[i]['right_anchor'] + shifts[i]
        
    # 5. Assemble output
    total_width = sum(mo['width'] for mo in meta_objects)
    out = np.zeros((h, total_width), dtype=int)
    
    curr_c = 0
    for i, mo in enumerate(meta_objects):
        shift = shifts[i]
        for c in range(mo['width']):
            for r in range(h):
                val = mo['cols'][r, c]
                if val != 0:
                    new_r = r + shift
                    if 0 <= new_r < h:
                        out[new_r, curr_c + c] = val
        curr_c += mo['width']
        
    return out

def solve_connect_meta_objects(solver) -> Optional[List[np.ndarray]]:
    """
    Groups non-empty columns into meta-objects, recolors markers (5), and joins them 
    by aligning anchors (first marker or first non-zero in boundary columns).
    """
    for i, (inp, out) in enumerate(solver.pairs):
        pred = solve_one(inp)
        if pred.shape != out.shape or not np.array_equal(pred, out):
            return None
    return [solve_one(ti) for ti in solver.test_in]
