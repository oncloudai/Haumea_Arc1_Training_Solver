import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_global_stencil_tiling(solver) -> Optional[List[np.ndarray]]:
    bg = 0
    
    def get_nucleus(grid):
        # A nucleus is a connected component of pixels where each color is unique (or mostly unique)
        # Actually, let's just find small components.
        labeled, n = label(grid != bg, structure=np.ones((3,3)))
        nuclei = []
        for i in range(1, n + 1):
            coords = np.argwhere(labeled == i)
            if len(coords) < 10: # Small enough to be a nucleus
                nuclei.append(coords)
        return nuclei

    def get_stencil(grid):
        # A stencil is a large connected component that serves as a template
        labeled, n = label(grid != bg, structure=np.ones((3,3)))
        stencils = []
        for i in range(1, n+1):
            coords = np.argwhere(labeled == i)
            if len(coords) >= 4: # Large enough
                # Normalize coords to relative
                min_r, min_c = coords.min(axis=0)
                rel = sorted([(r - min_r, c - min_c) for r, c in coords])
                # Find color(s) of this stencil in the nucleus?
                # Actually, maybe the stencil IS the multi-colored object.
                stencils.append({'rel': rel, 'coords': coords, 'color': grid[coords[0][0], coords[0][1]]})
        return stencils

    def process(grid):
        h, w = grid.shape
        res = grid.copy()
        
        nuclei = get_nucleus(grid)
        stencils = get_stencil(grid)
        if not nuclei or not stencils: return grid
        
        # Use the first stencil found
        stencil = stencils[0]
        s_rel = stencil['rel']
        
        for n_coords in nuclei:
            # For each pixel in nucleus, stamp the stencil reflected/recolored?
            # Or maybe just stamp the stencil with the nucleus's color?
            
            # Find the "center" of the nucleus
            min_rn, min_cn = n_coords.min(axis=0)
            max_rn, max_cn = n_coords.max(axis=0)
            
            # For each pixel in nucleus
            for r_n, c_n in n_coords:
                color = grid[r_n, c_n]
                # Reflection based on position in nucleus
                h_flip = (c_n > min_cn)
                v_flip = (r_n > min_rn)
                
                for dr, dc in s_rel:
                    nr, nc = dr, dc
                    if h_flip: nc = -nc
                    if v_flip: nr = -nr
                    
                    fr, fc = r_n + nr, c_n + nc
                    if 0 <= fr < h and 0 <= fc < w:
                        if res[fr, fc] == bg: res[fr, fc] = color
        return res

    consistent = True
    found_any = False
    for inp, out in solver.pairs:
        pred = process(inp)
        if not np.array_equal(pred, out): consistent = False; break
        if not np.array_equal(pred, inp): found_any = True
    
    if consistent and found_any:
        return [process(ti) for ti in solver.test_in]
    return None
