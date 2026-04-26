import numpy as np
from typing import List, Optional
from solver.utils import get_blobs

def solve_template_periodicity(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies the largest blob as a template.
    For each other blob (marker), propagates the template shape in the direction 
    of the marker. The period is max(template_height, template_width) + 1.
    """
    consistent = True; found_any = False
    
    for inp, out in solver.pairs:
        h, w = inp.shape
        blobs = get_blobs(inp, background=0, connectivity=8)
        if not blobs: consistent = False; break
        
        template_blob = max(blobs, key=lambda b: b['size'])
        t_min = template_blob['coords'].min(axis=0)
        t_max = template_blob['coords'].max(axis=0)
        t_center = (t_min + t_max) // 2
        template_shape_rel = template_blob['coords'] - t_center
        template_dim = max(t_max - t_min) + 1
        period = template_dim + 1
        
        pred = inp.copy()
        for b in blobs:
            if b is template_blob: continue
            
            m_center = (b['coords'].min(axis=0) + b['coords'].max(axis=0)) // 2
            v = m_center - t_center
            if np.all(v == 0): continue
            
            step_r = np.sign(v[0]) * period if v[0] != 0 else 0
            step_c = np.sign(v[1]) * period if v[1] != 0 else 0
            
            for k in range(0, max(h, w)):
                nr, nc = t_center[0] + k * step_r, t_center[1] + k * step_c
                if nr < -period or nr > h + period or nc < -period or nc > w + period:
                    break
                for pr, pc in template_shape_rel:
                    tr, tc = nr + pr, nc + pc
                    if 0 <= tr < h and 0 <= tc < w:
                        if pred[tr, tc] == 0:
                            pred[tr, tc] = b['color']
                            found_any = True
        
        if not np.array_equal(pred, out):
            consistent = False; break

    if consistent and found_any:
        results = []
        for ti in solver.test_in:
            h, w = ti.shape; res = ti.copy()
            blobs = get_blobs(ti, background=0, connectivity=8)
            if not blobs:
                results.append(res); continue
            
            template_blob = max(blobs, key=lambda b: b['size'])
            t_min = template_blob['coords'].min(axis=0); t_max = template_blob['coords'].max(axis=0)
            t_center = (t_min + t_max) // 2
            template_shape_rel = template_blob['coords'] - t_center
            template_dim = max(t_max - t_min) + 1
            period = template_dim + 1
            
            for b in blobs:
                if b is template_blob: continue
                m_center = (b['coords'].min(axis=0) + b['coords'].max(axis=0)) // 2
                v = m_center - t_center
                if np.all(v == 0): continue
                step_r = np.sign(v[0]) * period if v[0] != 0 else 0
                step_c = np.sign(v[1]) * period if v[1] != 0 else 0
                for k in range(0, max(h, w)):
                    nr, nc = t_center[0] + k * step_r, t_center[1] + k * step_c
                    if nr < -period or nr > h + period or nc < -period or nc > w + period: break
                    for pr, pc in template_shape_rel:
                        tr, tc = nr + pr, nc + pc
                        if 0 <= tr < h and 0 <= tc < w:
                            if res[tr, tc] == 0: res[tr, tc] = b['color']
            results.append(res)
        return results
    return None
