import numpy as np
from typing import List, Optional

def solve_fill_gaps_between_supported_runs(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies horizontal runs of color 2.
    Checks if runs have 'vertical support' (adjacent rows also have 2s in that range).
    Fills gaps between such runs with color 9 if no intermediate structure is collapsing.
    """
    def run_has_vertical_support(grid, r, start, end):
        h = grid.shape[0]
        for rr in [r-1, r+1]:
            if 0 <= rr < h:
                if np.all(grid[rr, start:end+1] == 2):
                    return True
        return False

    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        out = grid.copy()
        
        for r in range(h):
            runs = []
            c = 0
            while c < w:
                if grid[r, c] == 2:
                    start = c
                    while c < w and grid[r, c] == 2: c += 1
                    end = c - 1
                    runs.append((start, end))
                else: c += 1
            
            if len(runs) >= 2:
                left_run = runs[0]
                right_run = runs[-1]
                
                if (run_has_vertical_support(grid, r, *left_run) and
                    run_has_vertical_support(grid, r, *right_run)):
                    
                    gap_start, gap_end = left_run[1] + 1, right_run[0] - 1
                    above_has_2s = (r > 0 and np.any(grid[r-1, gap_start:gap_end+1] == 2))
                    below_has_2s = (r < h-1 and np.any(grid[r+1, gap_start:gap_end+1] == 2))
                    current_has_2s = np.any(grid[r, gap_start:gap_end+1] == 2)
                    
                    structure_collapsing = (not current_has_2s) and (above_has_2s or below_has_2s)
                    
                    if not structure_collapsing:
                        for col in range(gap_start, gap_end + 1):
                            if out[r, col] == 0: out[r, col] = 9
        return out

    for inp, out in solver.pairs:
        if not np.array_equal(apply_logic(inp), out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
