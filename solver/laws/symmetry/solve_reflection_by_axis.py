import numpy as np
from typing import List, Optional

def solve_reflection_by_axis(solver) -> Optional[List[np.ndarray]]:
    """
    Finds a target color and an axis (vertical or horizontal) such that
    reflecting pixels of the target color across the axis results in the output.
    """
    input_colors = []
    for inp, out in solver.pairs:
        input_colors.extend(np.unique(inp).tolist())
    input_colors = sorted(list(set(input_colors)))
    
    # print(f"DEBUG: input_colors={input_colors}")
    
    # Target color can be any color present in input (usually non-zero)
    for target_color in input_colors:
        if target_color == 0: continue
        
        # 1. Try vertical axes (c_ref = 2*axis_c - c)
        W = solver.pairs[0][0].shape[1]
        for axis_c in np.arange(0, W, 0.5):
            consistent = True
            found_change = False
            for pair_idx, (inp, out) in enumerate(solver.pairs):
                if inp.shape != out.shape:
                    consistent = False; break
                
                pred = inp.copy()
                target_coords = np.argwhere(inp == target_color)
                if target_coords.size == 0:
                    if not np.array_equal(inp, out):
                        consistent = False; break
                    continue
                
                for r, c in target_coords:
                    c_ref = int(round(2 * axis_c - c))
                    if 0 <= c_ref < inp.shape[1]:
                        pred[r, c] = inp[r, c_ref]
                    else:
                        pred[r, c] = 0
                    found_change = True
                
                if not np.array_equal(pred, out):
                    # if target_color == 1 and axis_c == 5.5:
                    #     print(f"DEBUG: target=1, axis=5.5 FAILED at pair {pair_idx}")
                    consistent = False; break
            
            if consistent and found_change:
                # print(f"DEBUG: target={target_color}, axis={axis_c} SUCCESS!")
                results = []
                for ti in solver.test_in:
                    res = ti.copy()
                    for r, c in np.argwhere(ti == target_color):
                        c_ref = int(round(2 * axis_c - c))
                        if 0 <= c_ref < ti.shape[1]:
                            res[r, c] = ti[r, c_ref]
                        else:
                            res[r, c] = 0
                    results.append(res)
                return results
    return None
