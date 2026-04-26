import numpy as np
from typing import List, Optional
from solver.utils import get_holes

def solve_enclosure_fill(solver) -> Optional[List[np.ndarray]]:
    # Strategy 1: Background component check
    for inp, out in solver.pairs:
        hls = get_holes(inp)
        if not hls: continue
        unq_in = np.unique(inp); unq_out = np.unique(out)
        new_colors = [c for c in unq_out if c not in unq_in]
        if not new_colors: continue
        fill_c = new_colors[0]
        def process(grid, fc):
            res = grid.copy()
            for h in get_holes(grid):
                for r, c in h['coords']: res[r, c] = fc
            return res
        if np.array_equal(process(inp, fill_c), out):
            if all(np.array_equal(process(i, fill_c), o) for i, o in solver.pairs):
                return [process(ti, fill_c) for ti in solver.test_in]
    return None
