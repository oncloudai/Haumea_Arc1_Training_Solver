import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_guide_and_shape_stamping(solver) -> Optional[List[np.ndarray]]:
    """
    1. Identifies two vertical 'guides' defined by yellow (4) markers at their ends.
    2. Learns the guide colors G1 and G2 from the segments between markers.
    3. Finds all pixels of colors G1 and G2 that are NOT part of the guide segments.
    4. These pixels form an 'object' whose bounding box determines the pattern.
    5. The output is the region between the guides, filled by the object pattern.
    6. If the horizontal order of G1 and G2 in the object is different from the 
       order of the guide columns, the object pattern is flipped horizontally.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        # 1. Find guide columns and markers
        guides = []
        for c in range(w):
            y_rows = np.where(grid[:, c] == 4)[0]
            if len(y_rows) == 2:
                r1, r2 = y_rows[0], y_rows[1]
                mid_segment = grid[r1+1:r2, c]
                unique_mid = np.unique(mid_segment)
                colors = [m for m in unique_mid if m != 0]
                if len(colors) == 1:
                    guides.append({
                        'color': colors[0],
                        'col': c,
                        'r1': r1,
                        'r2': r2,
                        'segment': set([(r, c) for r in range(r1+1, r2)])
                    })
        
        if len(guides) < 2: return None
        guides.sort(key=lambda x: x['col'])
        g1, g2 = guides[0], guides[1]
        
        gr1, gr2 = g1['r1'], g1['r2']
        gc1, gc2 = g1['col'], g2['col']
        
        # 2. Find the object O (all G1, G2 pixels not in guides)
        o_pixels = []
        g1_o_cols = []
        g2_o_cols = []
        
        for r in range(h):
            for c in range(w):
                color = grid[r, c]
                if color == g1['color'] or color == g2['color']:
                    if (r, c) not in g1['segment'] and (r, c) not in g2['segment']:
                        o_pixels.append((r, c, color))
                        if color == g1['color']: g1_o_cols.append(c)
                        if color == g2['color']: g2_o_cols.append(c)
                        
        if not o_pixels: return None
        
        or_coords = [p[0] for p in o_pixels]
        oc_coords = [p[1] for p in o_pixels]
        or1, or2 = min(or_coords), max(or_coords)
        oc1, oc2 = min(oc_coords), max(oc_coords)
        
        # 3. Check for flip
        # Guide order is G1 then G2 (since gc1 < gc2)
        # Object order:
        g1_left = min(g1_o_cols) if g1_o_cols else float('inf')
        g2_left = min(g2_o_cols) if g2_o_cols else float('inf')
        
        needs_flip = False
        if g1_left > g2_left:
            needs_flip = True
            
        # 4. Construct output
        out_h = gr2 - gr1 + 1
        out_w = gc2 - gc1 + 1
        out = np.zeros((out_h, out_w), dtype=int)
        
        # Yellow markers
        out[0, 0] = out[0, out_w-1] = out[out_h-1, 0] = out[out_h-1, out_w-1] = 4
        
        # Guide segments
        for r in range(1, out_h - 1):
            out[r, 0] = g1['color']
            out[r, out_w - 1] = g2['color']
            
        # Object pattern
        # We need to map O's rows [or1, or2] to output rows [1, out_h-2]
        # and O's columns [oc1, oc2] to output columns [1, out_w-2]
        
        obj_h = or2 - or1 + 1
        obj_w = oc2 - oc1 + 1
        
        for r, c, color in o_pixels:
            # Map to relative [0, obj_h-1] and [0, obj_w-1]
            dr = r - or1
            dc = c - oc1
            
            if needs_flip:
                dc = (obj_w - 1) - dc
                # If we flipped, the color might need to be swapped if we are 
                # treating them as separate shapes? 
                # No, the user says "flip to follow same color".
                # This implies if we flip the shape, G1 pixels move to where 
                # they align with G1 guide.
                
            # Map to output (offset by 1 for guides)
            nr = dr + 1
            nc = dc + 1
            
            if 0 <= nr < out_h and 0 <= nc < out_w:
                out[nr, nc] = color
                
        return out

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
            
    results = []
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
