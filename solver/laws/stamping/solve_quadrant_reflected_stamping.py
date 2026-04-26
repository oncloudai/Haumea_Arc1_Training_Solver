import numpy as np
from typing import List, Optional

def solve_quadrant_reflected_stamping(solver) -> Optional[List[np.ndarray]]:
    """
    Handle multiple templates and markers. Match each marker to a template by color.
    Stamp the template reflected according to the marker's quadrant.
    """
    consistent = True
    for pair_idx, (inp, out) in enumerate(solver.pairs):
        res = inp.copy()
        h, w = inp.shape
        mid_r, mid_c = h / 2, w / 2
        
        # Find all objects
        labeled = np.zeros_like(inp)
        num_objects = 0
        all_coords = np.argwhere(inp != 0)
        for r, c in all_coords:
            if labeled[r, c] == 0:
                num_objects += 1
                q = [(r, c)]
                labeled[r, c] = num_objects
                while q:
                    curr_r, curr_c = q.pop(0)
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < h and 0 <= nc < w and inp[nr, nc] != 0 and labeled[nr, nc] == 0:
                                labeled[nr, nc] = num_objects
                                q.append((nr, nc))
        
        templates = []; markers = []
        for obj_id in range(1, num_objects + 1):
            coords = np.argwhere(labeled == obj_id)
            if len(coords) == 1:
                markers.append(coords[0])
            else:
                templates.append(obj_id)
                
        if not templates or not markers: 
            consistent = False; break
        
        # For each marker, find its corresponding template
        for mr, mc in markers:
            m_color = inp[mr, mc]
            res[mr, mc] = 0 # Clear marker
            
            target_template = None
            anchor = None
            for tid in templates:
                coords = np.argwhere(labeled == tid)
                for r, c in coords:
                    if inp[r, c] == m_color:
                        target_template = tid
                        anchor = (r, c)
                        break
                if target_template: break
                
            if not target_template:
                consistent = False; break
                
            template_coords = np.argwhere(labeled == target_template)
            template_rel = [(r - anchor[0], c - anchor[1], inp[r, c]) for r, c in template_coords]
            
            ar_side = anchor[0] >= mid_r
            ac_side = anchor[1] >= mid_c
            mr_side = mr >= mid_r
            mc_side = mc >= mid_c
            
            fv = -1 if mr_side != ar_side else 1
            fh = -1 if mc_side != ac_side else 1
            
            for dr, dc, color in template_rel:
                nr, nc = mr + dr * fv, mc + dc * fh
                if 0 <= nr < h and 0 <= nc < w:
                    res[nr, nc] = color
                    
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        results = []
        for ti in solver.test_in:
            h, w = ti.shape
            mid_r, mid_c = h / 2, w / 2
            labeled = np.zeros_like(ti); num_objects = 0
            all_coords = np.argwhere(ti != 0)
            for r, c in all_coords:
                if labeled[r, c] == 0:
                    num_objects += 1; q = [(r, c)]; labeled[r, c] = num_objects
                    while q:
                        curr_r, curr_c = q.pop(0)
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                nr, nc = curr_r + dr, curr_c + dc
                                if 0 <= nr < h and 0 <= nc < w and ti[nr, nc] != 0 and labeled[nr, nc] == 0:
                                    labeled[nr, nc] = num_objects; q.append((nr, nc))
            templates = []; markers = []
            for obj_id in range(1, num_objects + 1):
                coords = np.argwhere(labeled == obj_id)
                if len(coords) == 1: markers.append(coords[0])
                else: templates.append(obj_id)
            res = ti.copy()
            for mr, mc in markers:
                m_color = ti[mr, mc]; res[mr, mc] = 0; target_template = None; anchor = None
                for tid in templates:
                    coords = np.argwhere(labeled == tid)
                    for r, c in coords:
                        if ti[r, c] == m_color: target_template = tid; anchor = (r, c); break
                    if target_template: break
                if not target_template: continue
                template_coords = np.argwhere(labeled == target_template)
                template_rel = [(r - anchor[0], c - anchor[1], ti[r, c]) for r, c in template_coords]
                ar_side, ac_side = anchor[0] >= mid_r, anchor[1] >= mid_c
                mr_side, mc_side = mr >= mid_r, mc >= mid_c
                fv, fh = (-1 if mr_side != ar_side else 1), (-1 if mc_side != ac_side else 1)
                for dr, dc, color in template_rel:
                    nr, nc = mr + dr * fv, mc + dc * fh
                    if 0 <= nr < h and 0 <= nc < w: res[nr, nc] = color
            results.append(res)
        return results
    return None
