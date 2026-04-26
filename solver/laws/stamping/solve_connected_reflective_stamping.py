import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_connected_reflective_stamping(solver) -> Optional[List[np.ndarray]]:
    bg = 0
    
    def process(grid):
        h, w = grid.shape
        res = grid.copy()
        
        # 1. Find connected components
        labeled, n_comp = label(grid != bg, structure=np.ones((3,3)))
        
        for i in range(1, n_comp + 1):
            comp_mask = (labeled == i)
            comp_coords = np.argwhere(comp_mask)
            
            # Find colors in this component
            unq_colors = np.unique(grid[comp_mask])
            
            # Find master color (the one with most pixels in this component)
            counts = {c: np.sum((grid == c) & comp_mask) for c in unq_colors}
            master_color = max(counts, key=counts.get)
            
            if counts[master_color] <= 1: continue
            
            # Master shape and its nucleus/seed
            master_pixels = np.argwhere((grid == master_color) & comp_mask)
            # Seed of master is the one adjacent to other colors in the same component
            master_seed = None
            for r, c in master_pixels:
                is_adj = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if grid[nr, nc] != bg and grid[nr, nc] != master_color and comp_mask[nr, nc]:
                                is_adj = True; break
                    if is_adj: break
                if is_adj:
                    master_seed = (r, c); break
            
            if master_seed is None: continue
            
            # Other colors are targets
            template = [(r - master_seed[0], c - master_seed[1]) for r, c in master_pixels]
            
            for t_color in unq_colors:
                if t_color == master_color: continue
                target_pixels = np.argwhere((grid == t_color) & comp_mask)
                for tr, tc in target_pixels:
                    # Reflection based on target_seed - master_seed
                    h_flip = (tc != master_seed[1])
                    v_flip = (tr != master_seed[0])
                    
                    for dr, dc in template:
                        nr, nc = dr, dc
                        if h_flip: nc = -nc
                        if v_flip: nr = -nr
                        fr, fc = tr + nr, tc + nc
                        if 0 <= fr < h and 0 <= fc < w:
                            if res[fr, fc] == bg or res[fr, fc] == t_color:
                                res[fr, fc] = t_color
        return res

    consistent = True
    found_any = False
    for inp, out in solver.pairs:
        pred = process(inp)
        if not np.array_equal(pred, out):
            consistent = False; break
        if not np.array_equal(pred, inp):
            found_any = True
            
    if consistent and found_any:
        return [process(ti) for ti in solver.test_in]
    return None
