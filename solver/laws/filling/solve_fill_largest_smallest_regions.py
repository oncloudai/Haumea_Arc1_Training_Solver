
import numpy as np
from typing import List, Optional

def solve_fill_largest_smallest_regions(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies all connected regions of 0s (4-connectivity).
    Fills the largest region with color 1.
    Fills the smallest region(s) with color 8.
    All other regions stay 0.
    """
    def run_single(input_grid):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        output = grid.copy()
        
        visited = np.zeros_like(grid, dtype=bool)
        regions = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 0 and not visited[r, c]:
                    q = [(r, c)]
                    visited[r, c] = True
                    region = []
                    while q:
                        cr, cc = q.pop(0)
                        region.append((cr, cc))
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<rows and 0<=nc<cols and grid[nr,nc]==0 and not visited[nr,nc]:
                                visited[nr,nc]=True; q.append((nr,nc))
                    regions.append(region)
        
        if not regions: return None
        
        sizes = [len(r) for r in regions]
        max_size = max(sizes)
        min_size = min(sizes)
        
        # In this task, if max_size == min_size, maybe nothing happens or both?
        # But training has distinct max/min.
        if max_size == min_size:
            # Maybe return None to let other laws try?
            # Or fill both?
            # Let's check training.
            pass
            
        found_any = False
        for r in regions:
            s = len(r)
            if s == max_size:
                for pr, pc in r:
                    output[pr, pc] = 1
                found_any = True
            elif s == min_size:
                for pr, pc in r:
                    output[pr, pc] = 8
                found_any = True
                
        return output if found_any else None

    # Verify on pairs
    for inp, out_expected in solver.pairs:
        pred = run_single(inp)
        if pred is None or not np.array_equal(pred, out_expected):
            return None
            
    test_preds = []
    for inp in solver.test_in:
        pred = run_single(inp)
        test_preds.append(pred if pred is not None else np.array(inp))
        
    return test_preds
