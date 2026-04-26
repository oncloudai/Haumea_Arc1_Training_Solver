
import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def get_8_symmetries(pixels):
    if not pixels: return [[]]
    # pixels is list of (r, c, col)
    coords = np.array([(p[0], p[1]) for p in pixels])
    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    
    grid = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for r, c, col in pixels:
        grid[r - min_r, c - min_c] = col
        
    res_grids = []
    curr = grid
    for _ in range(4):
        res_grids.append(curr)
        res_grids.append(np.fliplr(curr))
        curr = np.rot90(curr)
        
    res = []
    for g in res_grids:
        g_pixels = []
        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                if g[r, c] != 0:
                    g_pixels.append((r, c, g[r, c]))
        res.append(g_pixels)
    return res

def solve_anchor_template_matching(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a drawing color and logical objects (templates) containing it.
    Templates also contain 'anchors' (other colors).
    Finds isolated 'hints' in the input and matches templates to them 
    using 8 symmetries and translation.
    """
    consistent = True; found_any = False
    
    def process(inp):
        h, w = inp.shape
        unique, counts = np.unique(inp, return_counts=True)
        if len(unique) <= 1: return None
        
        # drawing_color is the most frequent non-zero color
        color_freq = {c: count for c, count in zip(unique, counts) if c != 0}
        if not color_freq: return None
        drawing_color = max(color_freq, key=color_freq.get)
        
        mask = (inp != 0).astype(int)
        labeled, n = label(mask, structure=np.ones((3,3)))
        components = []
        for idx in range(1, n + 1):
            coords = np.argwhere(labeled == idx)
            pixels = [(r, c, inp[r, c]) for r, c in coords]
            components.append(pixels)
        
        templates = []
        all_template_pixels = set()
        for comp in components:
            if any(p[2] == drawing_color for p in comp):
                templates.append(comp)
                for p in comp: all_template_pixels.add((p[0], p[1]))
                
        hints = []
        for r in range(h):
            for c in range(w):
                if inp[r, c] != 0 and (r, c) not in all_template_pixels:
                    hints.append((r, c, inp[r, c]))
        
        pred = np.zeros_like(inp)
        for temp in templates:
            anchors = [p for p in temp if p[2] != drawing_color]
            if not anchors: continue
            
            found = False
            for sym_pixels in get_8_symmetries(temp):
                sym_anchors = [p for p in sym_pixels if p[2] != drawing_color]
                if not sym_anchors: continue
                
                a0_r, a0_c, a0_col = sym_anchors[0]
                for h_r, h_c, h_col in hints:
                    if h_col == a0_col:
                        dr, dc = h_r - a0_r, h_c - a0_c
                        match = True
                        for i in range(1, len(sym_anchors)):
                            ar, ac, acol = sym_anchors[i]
                            tr, tc = ar + dr, ac + dc
                            if not any(hr == tr and hc == tc and hcl == acol for hr, hc, hcl in hints):
                                match = False; break
                        
                        if match:
                            for r, c, col in sym_pixels:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < h and 0 <= nc < w:
                                    pred[nr, nc] = col
                            found = True; break
                if found: break
        return pred

    for inp, out in solver.pairs:
        p = process(inp)
        if p is None or not np.array_equal(p, out):
            consistent = False; break
        found_any = True
            
    if consistent and found_any:
        results = []
        for ti in solver.test_in:
            p = process(ti)
            if p is None: return None
            results.append(p)
        return results
    return None
