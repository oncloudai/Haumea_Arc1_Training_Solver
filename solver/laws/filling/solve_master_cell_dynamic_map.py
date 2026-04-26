import numpy as np
from typing import List, Optional

def solve_master_cell_dynamic_map(solver) -> Optional[List[np.ndarray]]:
    bg = 0
    def get_cell_grid(grid):
        h, w = grid.shape
        for d in range(1, 10):
            rows = np.where(np.all(grid == d, axis=1))[0]
            cols = np.where(np.all(grid == d, axis=0))[0]
            if len(rows) == 2 and len(cols) == 2:
                rb = [-1] + sorted(list(rows)) + [h]
                cb = [-1] + sorted(list(cols)) + [w]
                if (all(rb[i+1]-rb[i] == rb[1]-rb[0] for i in range(3)) and 
                    all(cb[i+1]-cb[i] == cb[1]-cb[0] for i in range(3))):
                    return d, rb, cb
        return None, None, None

    def find_master_cell(grid, rb, cb):
        centers = []
        for i in range(3):
            for j in range(3):
                cell = grid[rb[i]+1:rb[i+1], cb[j]+1:cb[j+1]]
                centers.append(int(cell[1, 1]))
        
        # Find unique non-zero center
        unq, counts = np.unique(centers, return_counts=True)
        unique_centers = [c for c, count in zip(unq, counts) if count == 1 and c != 0]
        
        if not unique_centers: return None
        
        # Priority: pick one (assuming only one exists based on hint)
        master_val = unique_centers[0]
        idx = centers.index(master_val)
        return idx // 3, idx % 3

    def process(grid):
        d, rb, cb = get_cell_grid(grid)
        if d is None: return None
        master_idx = find_master_cell(grid, rb, cb)
        if master_idx is None: return None
        mi, mj = master_idx
        
        master = grid[rb[mi]+1:rb[mi+1], cb[mj]+1:cb[mj+1]]
        res = grid.copy(); rh, cw = master.shape
        for r in range(3):
            for c in range(3):
                color = master[r, c]
                res[rb[r]+1:rb[r+1], cb[c]+1:cb[c+1]] = color
        return res

    for inp, out in solver.pairs:
        p = process(inp)
        if p is None or not np.array_equal(p, out): return None
        
    return [process(ti) for ti in solver.test_in]
