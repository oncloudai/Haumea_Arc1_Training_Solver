
import numpy as np
from typing import List, Optional

def solve_tiled_with_diagonal_markers(solver) -> Optional[List[np.ndarray]]:
    """
    Tiles the input 2x2.
    For every seed (non-zero) pixel in the tiled output, 
    adds color 8 at positions (r+/-1, c+/-1) if that position is 0
    AND does not correspond to a seed position in the tiling.
    """
    consistent = True; found_any = False
    
    def process(inp, ho, wo):
        hi, wi = inp.shape
        # Create tiled output with seeds
        res = np.zeros((ho, wo), dtype=int)
        seeds = []
        for r in range(ho):
            for c in range(wo):
                val = inp[r % hi, c % wi]
                if val != 0:
                    res[r, c] = val
                    seeds.append((r, c))
        
        # Add diagonal markers
        for sr, sc in seeds:
            for dr in [-1, 1]:
                for dc in [-1, 1]:
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < ho and 0 <= nc < wo:
                        # Check if it's a seed position in ANY tile
                        if inp[nr % hi, nc % wi] == 0:
                            res[nr, nc] = 8
        return res

    for inp, out in solver.pairs:
        hi, wi = inp.shape
        ho, wo = out.shape
        if ho != 2 * hi or wo != 2 * wi:
            consistent = False; break
        pred = process(inp, ho, wo)
        if not np.array_equal(pred, out):
            consistent = False; break
        found_any = True
            
    if consistent and found_any:
        results = []
        for ti in solver.test_in:
            hi, wi = ti.shape
            # Assuming 2x2 tiling for test as well
            ho, wo = 2 * hi, 2 * wi
            results.append(process(ti, ho, wo))
        return results
    return None
