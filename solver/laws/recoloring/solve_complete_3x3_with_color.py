
import numpy as np
from typing import List, Optional

def solve_complete_3x3_with_color(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies objects of a specific color. For each object, finds a 3x3 
    square that contains it and fills the empty (0) cells in that 3x3 
    with a new color.
    """
    def run_single(input_grid, source_color, fill_color):
        grid = np.array(input_grid)
        rows, cols = grid.shape
        output = grid.copy()
        
        labeled = np.zeros_like(grid)
        curr = 1
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == source_color and labeled[r, c] == 0:
                    q = [(r, c)]; labeled[r, c] = curr
                    while q:
                        cr, cc = q.pop(0)
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<rows and 0<=nc<cols and grid[nr,nc]==source_color and labeled[nr,nc]==0:
                                labeled[nr,nc]=curr; q.append((nr,nc))
                    curr += 1
        
        if curr == 1: return None
        
        found_any = False
        for i in range(1, curr):
            coords = np.argwhere(labeled == i)
            r_min, c_min = coords.min(axis=0); r_max, c_max = coords.max(axis=0)
            
            # Find a 3x3 that contains this object
            for r_start in range(max(0, r_max-2), min(rows-2, r_min+1)):
                for c_start in range(max(0, c_max-2), min(cols-2, c_min+1)):
                    # This 3x3 contains the object.
                    for dr in range(3):
                        for dc in range(3):
                            if output[r_start+dr, c_start+dc] == 0:
                                output[r_start+dr, c_start+dc] = fill_color
                                found_any = True
                    break
                else: continue
                break
        return output if found_any else None

    # Brute force search for (source_color, fill_color)
    for src in range(1, 10):
        for fill in range(1, 10):
            if src == fill: continue
            consistent = True
            found_change = False
            for inp, out in solver.pairs:
                pred = run_single(inp, src, fill)
                if pred is None or not np.array_equal(pred, out):
                    consistent = False; break
                found_change = True
            if consistent and found_change:
                return [run_single(ti, src, fill) if run_single(ti, src, fill) is not None else np.array(ti) for ti in solver.test_in]
                
    return None
