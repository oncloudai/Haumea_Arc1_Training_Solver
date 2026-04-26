
import numpy as np
from typing import List, Optional

def solve_object_overlay(solver) -> Optional[List[np.ndarray]]:
    """
    Extract each non-background object (connected component) and its bounding box.
    Overlay them into a single grid of the maximum observed size.
    """
    for bg_color in range(10):
        consistent = True
        for pair_idx, (inp, out) in enumerate(solver.pairs):
            h_out, w_out = out.shape
            # Simple object extraction: each color is its own layer?
            # Or connected components? Let's try colors first.
            colors = np.unique(inp); colors = colors[colors != bg_color]
            if not colors.tolist(): consistent = False; break
            
            res = np.full((h_out, w_out), bg_color)
            for c in colors:
                coords = np.argwhere(inp == c)
                if len(coords) == 0: continue
                r_min, c_min = coords[:,0].min(), coords[:,1].min()
                r_max, c_max = coords[:,0].max(), coords[:,1].max()
                obj = inp[r_min:r_max+1, c_min:c_max+1]
                
                # Where to place obj? Try all (r,c)?
                # Usually it's top-left, or such that it matches.
                # In 4290ef0e, they are overlaid into the output.
                # Try common alignments: top-left, center, etc.
                best_match = False
                for dr in range(h_out - obj.shape[0] + 1):
                    for dc in range(w_out - obj.shape[1] + 1):
                        temp = res.copy()
                        for r in range(obj.shape[0]):
                            for c_idx in range(obj.shape[1]):
                                if obj[r, c_idx] != bg_color:
                                    temp[dr+r, dc+c_idx] = obj[r, c_idx]
                        if np.all(temp == out): # Wait, we need to check AFTER ALL COLORS
                            pass
                # Overlaying is a bit complex. Let's try simple: each color at its relative offset?
                # No, they are disconnected in input.
                
                # Let's try: each color's bounding box is extracted and shifted to (0,0) of output?
                # No, they have different relative positions.
                pass
            
            # Simplified: output is exactly the overlay of non-background pixels 
            # if we align their local (r-rmin, c-cmin) to the output (r, c)?
            temp = np.full((h_out, w_out), bg_color)
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    if inp[r, c] != bg_color:
                        # Find which object this belongs to
                        pass
            
            # Let's try another heuristic: the output is a 7x7 grid, 
            # and each 7x7 subgrid in input is overlaid? No.
            
    # Actually, 4290ef0e is a "modular tiling" task. 
    # The output is a single cell of a larger implicit grid.
    return None
