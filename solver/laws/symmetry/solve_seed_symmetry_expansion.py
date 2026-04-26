import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_seed_symmetry_expansion(solver) -> Optional[List[np.ndarray]]:
    bg = 0
    
    def get_quadrant_pixels(grid, r_seed, c_seed, quad_type):
        color = grid[r_seed, c_seed]
        mask = (grid == color)
        if quad_type == 'TL': mask[r_seed+1:, :] = False; mask[:, c_seed+1:] = False
        elif quad_type == 'TR': mask[r_seed+1:, :] = False; mask[:, :c_seed] = False
        elif quad_type == 'BL': mask[:r_seed, :] = False; mask[:, c_seed+1:] = False
        elif quad_type == 'BR': mask[:r_seed, :] = False; mask[:, :c_seed] = False
        
        labeled, n = label(mask)
        if n == 0: return []
        target_label = labeled[r_seed, c_seed]
        if target_label == 0: return []
        
        coords = np.argwhere(labeled == target_label)
        return [(r - r_seed, c - c_seed) for r, c in coords]

    def apply_symmetry(grid):
        res = grid.copy()
        h, w = grid.shape
        
        # In Example 0, nuclei are not necessarily 2x2.
        # Let's find all pixels that have a neighbor of a different color.
        # These are potential seeds.
        seed_pixels = []
        for r in range(h):
            for c in range(w):
                if grid[r, c] == bg: continue
                # Check neighbors
                is_seed = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if grid[nr, nc] != bg and grid[nr, nc] != grid[r, c]:
                                is_seed = True; break
                    if is_seed: break
                if is_seed: seed_pixels.append((r, c))
        
        if not seed_pixels: return res
        
        # Group seeds into nuclei (connected components of seeds)
        seed_mask = np.zeros((h, w), dtype=bool)
        for r, c in seed_pixels: seed_mask[r, c] = True
        labeled_nuclei, n_nuclei = label(seed_mask, structure=np.ones((3,3)))
        
        for i in range(1, n_nuclei + 1):
            nucleus = np.argwhere(labeled_nuclei == i)
            # Find template within this nucleus
            template = None
            source_seed = None
            source_qt = None
            
            # For each pixel in nucleus, check if it's a source
            for r_s, c_s in nucleus:
                # Determine its quadrant type relative to the nucleus bounding box?
                # No, look at its neighbors within the nucleus.
                # Simplest: check all 4 quadrant types
                for qt in ['TL', 'TR', 'BL', 'BR']:
                    q_pix = get_quadrant_pixels(grid, r_s, c_s, qt)
                    if len(q_pix) > 1:
                        template = q_pix
                        source_seed = (r_s, c_s)
                        source_qt = qt
                        break
                if template: break
            
            if template:
                # Stamp recolored reflected template at every other seed in this nucleus
                for r_t, c_t in nucleus:
                    if r_t == source_seed[0] and c_t == source_seed[1]: continue
                    target_color = grid[r_t, c_t]
                    
                    # Determine transformation relative to source_seed
                    # In b775ac94, it's about H and V positions.
                    h_flip = (c_t != source_seed[1])
                    v_flip = (r_t != source_seed[0])
                    
                    for dr, dc in template:
                        nr, nc = dr, dc
                        if h_flip: nc = -nc
                        if v_flip: nr = -nr
                        fr, fc = r_t + nr, c_t + nc
                        if 0 <= fr < h and 0 <= fc < w:
                            if res[fr, fc] == bg: res[fr, fc] = target_color
        return res

    consistent = True; found_any = False
    for inp, out in solver.pairs:
        pred = apply_symmetry(inp)
        if not np.array_equal(pred, out): consistent = False; break
        if not np.array_equal(pred, inp): found_any = True
    if consistent and found_any:
        return [apply_symmetry(ti) for ti in solver.test_in]
    return None
