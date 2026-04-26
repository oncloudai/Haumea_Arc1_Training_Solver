import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_copy_stencil_to_empty_holes(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies 'stencil' components (hollow rectangular frames of a single color).
    Finds all other regions in the grid of the same size that are completely empty (all 0s).
    Copies the stencil pattern to each such empty region.
    """
    def is_hollow_rect(mask):
        rows, cols = np.where(mask)
        if len(rows) == 0: return False
        r1, r2 = rows.min(), rows.max()
        c1, c2 = cols.min(), cols.max()
        sh, sw = r2 - r1 + 1, c2 - c1 + 1
        if sh < 3 or sw < 3: return False
        
        # Must be exactly the boundary
        expected_count = 2 * sh + 2 * sw - 4
        if np.sum(mask) != expected_count: return False
        
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if r == r1 or r == r2 or c == c1 or c == c2:
                    if not mask[r, c]: return False
                else:
                    if mask[r, c]: return False
        return True

    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        out = grid.copy()
        
        stencils = []
        for color in range(1, 10):
            mask = (grid == color).astype(int)
            labeled, num_f = label(mask)
            for i in range(1, num_f + 1):
                comp = (labeled == i)
                if is_hollow_rect(comp):
                    rows, cols = np.where(comp)
                    r1, r2, c1, c2 = rows.min(), rows.max(), cols.min(), cols.max()
                    stencils.append({
                        'color': color,
                        'pattern': grid[r1:r2+1, c1:c2+1],
                        'h': r2-r1+1, 'w': c2-c1+1,
                        'orig_bbox': (r1, r2, c1, c2)
                    })
                    
        if not stencils: return None
        
        for s in stencils:
            sh, sw = s['h'], s['w']
            pattern = s['pattern']
            sr1, sr2, sc1, sc2 = s['orig_bbox']
            
            for r in range(h - sh + 1):
                for c in range(w - sw + 1):
                    if r == sr1 and c == sc1: continue
                    
                    # Check if region is all 0s
                    if np.all(grid[r:r+sh, c:c+sw] == 0):
                        # Copy pattern
                        for pr in range(sh):
                            for pc in range(sw):
                                if pattern[pr, pc] != 0:
                                    out[r + pr, c + pc] = pattern[pr, pc]
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
