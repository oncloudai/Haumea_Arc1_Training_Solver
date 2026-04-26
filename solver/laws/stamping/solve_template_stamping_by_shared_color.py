
import numpy as np
from scipy.ndimage import label

def get_all_blobs(grid):
    blobs = []
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not visited[r, c]:
                color = grid[r, c]
                labeled, n = label(grid == color)
                lbl = labeled[r, c]
                coords = np.argwhere(labeled == lbl)
                for cr, cc in coords: visited[cr, cc] = True
                r_min, c_min = coords.min(axis=0)
                r_max, c_max = coords.max(axis=0)
                blobs.append({
                    'color': int(color),
                    'coords': coords,
                    'bbox': (r_min, r_max, c_min, c_max),
                    'size': len(coords)
                })
    return blobs

def solve_template_stamping_by_shared_color(solver):
    def apply_logic(inp):
        inp = np.array(inp)
        h, w = inp.shape
        blobs = get_all_blobs(inp)
        blobs.sort(key=lambda x: x['size'], reverse=True)
        
        if len(blobs) < 3: return None
        
        large_bg = blobs[0]
        # Find medium blobs (backgrounds)
        medium_blobs = [b for b in blobs[1:] if b['size'] > 10]
        # Small blobs (patterns)
        small_blobs = [b for b in blobs[1:] if b['size'] <= 10]
        
        # Placeholders: small blobs inside medium blobs
        placeholders = []
        for s in small_blobs:
            for m in medium_blobs:
                p_rmin, p_rmax, p_cmin, p_cmax = m['bbox']
                if p_rmin <= s['bbox'][0] and s['bbox'][1] <= p_rmax and \
                   p_cmin <= s['bbox'][2] and s['bbox'][3] <= p_cmax:
                    placeholders.append(s)
                    break
        
        if not placeholders: return None
        
        # Template: small blobs inside large_bg
        template_candidates = []
        placeholder_ids = [id(p) for p in placeholders]
        for s in small_blobs:
            if id(s) not in placeholder_ids:
                template_candidates.append(s)
        
        if not template_candidates: return None
        
        # The template is the union of template_candidates that are "close" to each other
        # In these tasks, it's all of them.
        
        # For each placeholder, find if its color exists in the template
        outp = inp.copy()
        for s in placeholders:
            c = s['color']
            # Find a blob in template with color c
            anchor_t = None
            for t in template_candidates:
                if t['color'] == c:
                    anchor_t = t; break
            
            if anchor_t:
                # Replace s with the whole template
                # Center of s
                sr, sc = s['coords'][0] # Assume 1x1 for now
                # Center of anchor_t
                tr, tc = anchor_t['coords'][0]
                
                # Clear s
                for pr, pc in s['coords']: outp[pr, pc] = s['color'] # wait, no, clear it
                # Actually, in ARC we usually overwrite.
                
                for t in template_candidates:
                    # Offset of t relative to anchor_t
                    # wait, anchor_t might have multiple pixels. Use its first pixel.
                    for tpr, tpc in t['coords']:
                        dr, dc = tpr - tr, tpc - tc
                        nr, nc = sr + dr, sc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            outp[nr, nc] = t['color']
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
