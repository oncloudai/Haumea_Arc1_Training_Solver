import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def get_isometries():
    isometries = []
    # (r, c, h, w) -> (r', c')
    isometries.append(lambda r, c, h, w: (r, c)) # Id
    isometries.append(lambda r, c, h, w: (c, w - 1 - r)) # Rot 90
    isometries.append(lambda r, c, h, w: (h - 1 - r, w - 1 - c)) # Rot 180
    isometries.append(lambda r, c, h, w: (w - 1 - c, r)) # Rot 270
    isometries.append(lambda r, c, h, w: (h - 1 - r, c)) # Flip H
    isometries.append(lambda r, c, h, w: (r, w - 1 - c)) # Flip V
    isometries.append(lambda r, c, h, w: (c, r)) # Flip D1
    isometries.append(lambda r, c, h, w: (w - 1 - c, h - 1 - r)) # Flip D2
    return isometries

def solve_isometric_stamping(solver) -> Optional[List[np.ndarray]]:
    for anchor_color in range(1, 10):
        for bg in range(10):
            if anchor_color == bg: continue
            
            shapes = [] # List of {rel_coords, decorations, h, w, inp_decorations}
            consistent_task = True
            found_any_decoration = False
            
            for inp, out in solver.pairs:
                if inp.shape != out.shape: consistent_task = False; break
                
                # All anchor pixels
                anchor_mask = (inp == anchor_color)
                from scipy.ndimage import label
                labeled, n = label(anchor_mask)
                anchors = []
                for i in range(1, n+1):
                    coords = np.argwhere(labeled == i)
                    min_rc = coords.min(axis=0)
                    anchors.append({'coords': coords, 'top_left': min_rc})
                
                if not anchors: continue
                
                # Map every decoration (non-bg, non-anchor) to its closest anchor
                out_decorations = [] # (r, c, color)
                for r in range(out.shape[0]):
                    for c in range(out.shape[1]):
                        if out[r, c] != bg and out[r, c] != anchor_color:
                            out_decorations.append((r, c, out[r, c]))
                
                inp_decorations = []
                for r in range(inp.shape[0]):
                    for c in range(inp.shape[1]):
                        if inp[r, c] != bg and inp[r, c] != anchor_color:
                            inp_decorations.append((r, c, inp[r, c]))
                            
                for a in anchors:
                    r0, c0 = a['top_left']
                    ah, aw = a['coords'].max(axis=0) - r0 + 1
                    a_coords = set(map(tuple, a['coords']))
                    
                    local_out = {}
                    for ar, ac, color in out_decorations:
                        # Distance to this anchor
                        dists = np.sum(np.abs(a['coords'] - [ar, ac]), axis=1)
                        min_dist = np.min(dists)
                        # Check if this anchor is the closest one
                        is_closest = True
                        for other_a in anchors:
                            if other_a is a: continue
                            other_dists = np.sum(np.abs(other_a['coords'] - [ar, ac]), axis=1)
                            if np.min(other_dists) < min_dist:
                                is_closest = False; break
                        
                        if is_closest and min_dist <= 5:
                            local_out[(ar - r0, ac - c0)] = color
                            found_any_decoration = True
                            
                    local_inp = {}
                    for ar, ac, color in inp_decorations:
                        dists = np.sum(np.abs(a['coords'] - [ar, ac]), axis=1)
                        min_dist = np.min(dists)
                        is_closest = True
                        for other_a in anchors:
                            if other_a is a: continue
                            other_dists = np.sum(np.abs(other_a['coords'] - [ar, ac]), axis=1)
                            if np.min(other_dists) < min_dist:
                                is_closest = False; break
                        if is_closest and min_dist <= 5:
                            local_inp[(ar - r0, ac - c0)] = color

                    rel_coords = set((r - r0, c - c0) for r, c in a_coords)
                    shapes.append({
                        'rel_coords': rel_coords,
                        'out_decorations': local_out,
                        'inp_decorations': local_inp,
                        'h': ah, 'w': aw
                    })
            
            if not consistent_task or not found_any_decoration: continue
            
            templates = [] # List of {canonical_rel_coords, canonical_map, h, w}
            # canonical_map: { (dr, dc): {inp_color: out_color} }
            
            for s in shapes:
                found_template = False
                for t in templates:
                    for iso_func in get_isometries():
                        s_h, s_w = s['h'], s['w']
                        transformed_coords = set(iso_func(r, c, s_h, s_w) for r, c in s['rel_coords'])
                        tr_coords_arr = np.array(list(transformed_coords))
                        min_r, min_c = tr_coords_arr.min(axis=0)
                        normalized_tr_coords = set((r - min_r, c - min_c) for r, c in transformed_coords)
                        
                        if normalized_tr_coords == t['canonical_rel_coords']:
                            # Potential match. Check if decorations are consistent.
                            # We need to map relative offsets.
                            # For each offset (dr, dc) in out_decorations, what was in inp?
                            match_possible = True
                            for (dr, dc), out_color in s['out_decorations'].items():
                                tr_dr, tr_dc = iso_func(dr, dc, s_h, s_w)
                                norm_dr, norm_dc = tr_dr - min_r, tr_dc - min_c
                                inp_color = s['inp_decorations'].get((dr, dc), bg)
                                
                                if (norm_dr, norm_dc) in t['canonical_map']:
                                    if inp_color in t['canonical_map'][(norm_dr, norm_dc)]:
                                        if t['canonical_map'][(norm_dr, norm_dc)][inp_color] != out_color:
                                            match_possible = False; break
                                    else:
                                        t['canonical_map'][(norm_dr, norm_dc)][inp_color] = out_color
                                else:
                                    t['canonical_map'][(norm_dr, norm_dc)] = {inp_color: out_color}
                            
                            if match_possible:
                                found_template = True; break
                    if found_template: break
                
                if not found_template:
                    new_map = {}
                    for (dr, dc), out_color in s['out_decorations'].items():
                        inp_color = s['inp_decorations'].get((dr, dc), bg)
                        new_map[(dr, dc)] = {inp_color: out_color}
                    templates.append({
                        'canonical_rel_coords': s['rel_coords'],
                        'canonical_map': new_map,
                        'h': s['h'], 'w': s['w']
                    })

            # Verify
            all_train_correct = True
            for inp, out in solver.pairs:
                pred = inp.copy()
                from scipy.ndimage import label
                labeled, n = label(inp == anchor_color)
                anchors = [np.argwhere(labeled == i) for i in range(1, n+1)]
                for a_coords in anchors:
                    min_rc = a_coords.min(axis=0)
                    r0, c0 = min_rc
                    ah, aw = a_coords.max(axis=0) - r0 + 1
                    rel_coords = set((r - r0, c - c0) for r, c in a_coords)
                    
                    applied = False
                    for t in templates:
                        for iso_func in get_isometries():
                            transformed_coords = set(iso_func(r, c, t['h'], t['w']) for r, c in t['canonical_rel_coords'])
                            tr_coords_arr = np.array(list(transformed_coords))
                            t_min_r, t_min_c = tr_coords_arr.min(axis=0)
                            normalized_tr_coords = set((r - t_min_r, c - t_min_c) for r, c in transformed_coords)
                            
                            if normalized_tr_coords == rel_coords:
                                for (norm_dr, norm_dc), color_map in t['canonical_map'].items():
                                    # Inverse transform norm_dr, norm_dc? No, transform from canonical to current.
                                    # Wait, norm_dr is (tr_dr - t_min_r).
                                    # We need (target_dr, target_dc) relative to r0, c0.
                                    # This is just the isometry of some (dr, dc) in canonical.
                                    # But the map is already relative to canonical.
                                    # Let's re-isometry the (dr, dc) from canonical map.
                                    # dr_canon, dc_canon = offset in canonical
                                    # dr_curr, dc_curr = iso(dr_canon, dc_canon) - iso_min
                                    
                                    # Actually, let's just use the canonical offset and transform it.
                                    pass # will fix below
                                
                                # Correct way:
                                for (dr_c, dc_c), color_map in t['canonical_map'].items():
                                    tr_dr, tr_dc = iso_func(dr_c, dc_c, t['h'], t['w'])
                                    target_dr, target_dc = tr_dr - t_min_r, tr_dc - t_min_c
                                    target_r, target_c = r0 + target_dr, c0 + target_dc
                                    if 0 <= target_r < pred.shape[0] and 0 <= target_c < pred.shape[1]:
                                        inp_color = inp[target_r, target_c]
                                        if inp_color in color_map:
                                            pred[target_r, target_c] = color_map[inp_color]
                                applied = True; break
                        if applied: break
                if not np.array_equal(pred, out): all_train_correct = False; break
            
            if all_train_correct:
                results = []
                for ti in solver.test_in:
                    pred = ti.copy()
                    labeled, n = label(ti == anchor_color)
                    for i in range(1, n+1):
                        a_coords = np.argwhere(labeled == i)
                        r0, c0 = a_coords.min(axis=0)
                        rel_coords = set((r - r0, c - c0) for r, c in a_coords)
                        applied = False
                        for t in templates:
                            for iso_func in get_isometries():
                                transformed_coords = set(iso_func(r, c, t['h'], t['w']) for r, c in t['canonical_rel_coords'])
                                tr_coords_arr = np.array(list(transformed_coords))
                                t_min_r, t_min_c = tr_coords_arr.min(axis=0)
                                if set((r - t_min_r, c - t_min_c) for r, c in transformed_coords) == rel_coords:
                                    for (dr_c, dc_c), color_map in t['canonical_map'].items():
                                        tr_dr, tr_dc = iso_func(dr_c, dc_c, t['h'], t['w'])
                                        tr_r, tr_c = r0 + tr_dr - t_min_r, c0 + tr_dc - t_min_c
                                        if 0 <= tr_r < pred.shape[0] and 0 <= tr_c < pred.shape[1]:
                                            ic = ti[tr_r, tr_c]
                                            if ic in color_map: pred[tr_r, tr_c] = color_map[ic]
                                    applied = True; break
                            if applied: break
                    results.append(pred)
                return results
    return None
