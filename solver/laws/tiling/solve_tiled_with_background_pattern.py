import numpy as np
from typing import List, Optional

def solve_tiled_with_background_pattern(solver) -> Optional[List[np.ndarray]]:
    for bg_color in range(1, 10):
        rules = [
            lambda g: (np.indices(g.shape)[0] + np.indices(g.shape)[1]) % 2 == 0,
            lambda g: (np.indices(g.shape)[0] + np.indices(g.shape)[1]) % 2 == 1,
            lambda g: np.ones(g.shape, dtype=bool),
            lambda g: g == 0,
        ]
        
        for rule in rules:
            all_match = True
            for inp, out in solver.pairs:
                h, w = inp.shape
                if out.shape[0] != 2*h or out.shape[1] != 2*w: all_match = False; break
                p_mask = rule(inp)
                pred = np.tile(inp, (2, 2))
                t_mask = np.tile(p_mask, (2, 2))
                pred[(t_mask) & (pred == 0)] = bg_color
                if not np.array_equal(pred, out): all_match = False; break
            if all_match:
                return [np.where((np.tile(rule(ti), (2, 2))) & (np.tile(ti, (2, 2)) == 0), bg_color, np.tile(ti, (2, 2))) for ti in solver.test_in]
    return None
