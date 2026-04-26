
import numpy as np
from typing import List, Optional

def solve_row_repetition_extension(solver) -> Optional[List[np.ndarray]]:
    """
    Extend the grid vertically by repeating the rows of the input grid.
    Typically used when input height is small (e.g., 5 or 6) and output is 10.
    """
    def apply_logic(inp, out_h, out_w):
        inp_h, inp_w = inp.shape
        if inp_w != out_w: return None
        out = np.zeros((out_h, out_w), dtype=int)
        for r in range(out_h):
            out[r] = inp[r % inp_h]
        return out

    # Check if all train outputs have the same shape
    out_shapes = [out.shape for out in solver.train_out]
    if len(set(out_shapes)) != 1:
        # If not, let's try to see if each out.shape matches the logic
        pass
    
    for inp, out in solver.pairs:
        pred = apply_logic(inp, out.shape[0], out.shape[1])
        if pred is None:
            print(f"Pred is None for inp shape {inp.shape}")
            return None
        if not np.array_equal(pred, out):
            print(f"Pred mismatch for inp shape {inp.shape}")
            print("PRED:")
            print(pred)
            print("ACTUAL:")
            print(out)
            return None
            
    results = []
    # Use the height of the first training output for the test case if they are consistent
    target_h = solver.train_out[0].shape[0]
    for ti in solver.test_in:
        res = apply_logic(ti, target_h, ti.shape[1])
        if res is None: return None
        results.append(res)
    return results
