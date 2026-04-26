import numpy as np
from typing import List, Optional
from solver.utils import get_enclosed_holes

def solve_enclosure_fill_v2(solver) -> Optional[List[np.ndarray]]:
    # Strategy 2: binary_fill_holes on boundary colors
    for inp, out in solver.pairs:
        unq_in = np.unique(inp); unq_out = np.unique(out)
        new_colors = [c for c in unq_out if c not in unq_in]
        if not new_colors: continue
        fill_c = new_colors[0]
        for boundary_c in unq_in:
            if boundary_c == 0: continue
            def process2(grid, bc, fc):
                res = grid.copy()
                for h in get_enclosed_holes(grid, bc):
                    for r, c in h['coords']: res[r, c] = fc
                return res
            if np.array_equal(process2(inp, boundary_c, fill_c), out):
                if all(np.array_equal(process2(i, boundary_c, fill_c), o) for i, o in solver.pairs):
                    return [process2(ti, boundary_c, fill_c) for ti in solver.test_in]
    return None
