import numpy as np
from typing import List, Optional

def solve_move_color_towards_target_color(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a source pixel (color A) and a target pixel (color B).
    Moves the source pixel one step closer to the target pixel along both axes.
    In task dc433765: moves color 3 towards color 4.
    """
    def apply_logic(grid):
        grid = np.array(grid)
        h, w = grid.shape
        
        # We need to find the source color and target color
        # Since this is a law, we infer them from the first training pair
        return None

    def try_infer_params(inp, out):
        unique_in = np.unique(inp)
        unique_out = np.unique(out)
        
        # Source color is one that moved (was in in, position changed in out)
        # Target color is one that stayed put but served as a target
        candidates_source = [c for c in unique_in if c != 0]
        candidates_target = [c for c in unique_in if c != 0]
        
        for sc in candidates_source:
            for tc in candidates_target:
                if sc == tc: continue
                
                # Find positions in inp
                pos_s = np.argwhere(inp == sc)
                pos_t = np.argwhere(inp == tc)
                
                if len(pos_s) == 1 and len(pos_t) == 1:
                    r3, c3 = pos_s[0]
                    r4, c4 = pos_t[0]
                    
                    new_r, new_c = r3, c3
                    if r3 < r4: new_r = r3 + 1
                    elif r3 > r4: new_r = r3 - 1
                    if c3 < c4: new_c = c3 + 1
                    elif c3 > c4: new_c = c3 - 1
                    
                    test_out = inp.copy()
                    test_out[r3, c3] = 0
                    test_out[new_r, new_c] = sc
                    
                    if np.array_equal(test_out, out):
                        return (sc, tc)
        return None

    params = try_infer_params(solver.train_in[0], solver.train_out[0])
    if params is None: return None
    sc, tc = params
    
    def apply(grid):
        grid = np.array(grid)
        h, w = grid.shape
        out = grid.copy()
        
        pos_s = np.argwhere(grid == sc)
        pos_t = np.argwhere(grid == tc)
        
        if len(pos_s) == 1 and len(pos_t) == 1:
            r3, c3 = pos_s[0]
            r4, c4 = pos_t[0]
            
            new_r, new_c = r3, c3
            if r3 < r4: new_r = r3 + 1
            elif r3 > r4: new_r = r3 - 1
            if c3 < c4: new_c = c3 + 1
            elif c3 > c4: new_c = c3 - 1
            
            out[r3, c3] = 0
            out[new_r, new_c] = sc
        return out

    for inp, out_expected in solver.pairs:
        if not np.array_equal(apply(inp), out_expected):
            return None
            
    return [apply(ti) for ti in solver.test_in]
