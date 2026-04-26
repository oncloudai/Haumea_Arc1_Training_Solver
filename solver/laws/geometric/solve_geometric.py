import numpy as np
from typing import List, Optional

def solve_geometric(solver) -> Optional[List[np.ndarray]]:
    ops = [lambda g: g, lambda g: np.rot90(g, 1), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 3),
           lambda g: np.flipud(g), lambda g: np.fliplr(g), lambda g: np.transpose(g)]
    for op in ops:
        mapping = {}; consistent = True; found_change = False
        for inp, out in solver.pairs:
            t_inp = op(inp)
            if t_inp.shape != out.shape: consistent = False; break
            # Derive mapping
            for s, d in zip(t_inp.flatten(), out.flatten()):
                if s in mapping and mapping[s] != d: consistent = False; break
                mapping[s] = d
            if not consistent: break
            # Verify and check for change
            pred = np.vectorize(lambda x: mapping.get(x, x))(t_inp)
            if not np.array_equal(pred, out): consistent = False; break
            if not np.array_equal(pred, inp): found_change = True
        if consistent and found_change:
            return [np.vectorize(lambda x: mapping.get(x, x))(op(ti)) for ti in solver.test_in]
    return None
