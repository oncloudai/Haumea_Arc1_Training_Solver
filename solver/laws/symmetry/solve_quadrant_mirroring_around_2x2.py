
import numpy as np
from typing import List, Optional

def solve_quadrant_mirroring_around_2x2(solver) -> Optional[List[np.ndarray]]:
    """
    Finds a 2x2 block of a specific color (e.g., green color 3) that acts as a center.
    For each pixel of another specific color (e.g., red color 2), reflects it 
    across the rows and columns of the 2x2 center block to fill all four quadrants.
    """
    # Parameters to search: center_color, pattern_color
    for center_color in range(1, 10):
        for pattern_color in range(1, 10):
            if center_color == pattern_color: continue
            
            consistent = True
            found_any = False
            
            for inp, out in solver.pairs:
                grid = np.array(inp)
                h, w = grid.shape
                res = grid.copy()
                
                # 1. Find 2x2 center
                c_coords = np.argwhere(grid == center_color)
                if len(c_coords) != 4:
                    consistent = False; break
                gr1, gc1 = c_coords.min(axis=0)
                gr2, gc2 = c_coords.max(axis=0)
                if gr2 != gr1 + 1 or gc2 != gc1 + 1:
                    consistent = False; break
                
                # 2. Find pattern pixels
                p_coords = np.argwhere(grid == pattern_color)
                if len(p_coords) == 0:
                    # In some examples, the input might already be full or empty?
                    # But for a law to be valid, it should probably do something.
                    pass
                
                local_found = False
                for r, c in p_coords:
                    r_alt = (gr2 + (gr1 - r)) if r < gr1 else (gr1 - (r - gr2))
                    c_alt = (gc2 + (gc1 - c)) if c < gc1 else (gc1 - (c - gc2))
                    
                    for pr, pc in [(r, c), (r_alt, c), (r, c_alt), (r_alt, c_alt)]:
                        if 0 <= pr < h and 0 <= pc < w:
                            if res[pr, pc] != pattern_color:
                                res[pr, pc] = pattern_color
                                local_found = True
                
                if not np.array_equal(res, out):
                    consistent = False; break
                if local_found: found_any = True
                
            if consistent and found_any:
                # Successfully found the law! Apply to test inputs.
                results = []
                for ti in solver.test_in:
                    grid = np.array(ti)
                    h, w = grid.shape
                    res = grid.copy()
                    c_coords = np.argwhere(grid == center_color)
                    if len(c_coords) == 4:
                        gr1, gc1 = c_coords.min(axis=0)
                        gr2, gc2 = c_coords.max(axis=0)
                        if gr2 == gr1 + 1 and gc2 == gc1 + 1:
                            p_coords = np.argwhere(grid == pattern_color)
                            for r, c in p_coords:
                                r_alt = (gr2 + (gr1 - r)) if r < gr1 else (gr1 - (r - gr2))
                                c_alt = (gc2 + (gc1 - c)) if c < gc1 else (gc1 - (c - gc2))
                                for pr, pc in [(r, c), (r_alt, c), (r, c_alt), (r_alt, c_alt)]:
                                    if 0 <= pr < h and 0 <= pc < w:
                                        res[pr, pc] = pattern_color
                    results.append(res)
                return results
                
    return None
