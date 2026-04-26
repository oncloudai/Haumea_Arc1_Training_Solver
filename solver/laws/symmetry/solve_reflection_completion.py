import numpy as np
from typing import List, Optional

def solve_reflection_completion(solver) -> Optional[List[np.ndarray]]:
    for op in [np.flipud, np.fliplr, lambda x: np.transpose(x)]:
        consistent = True; found_change = False
        for inp, out in solver.pairs:
            try:
                pred = np.maximum(inp, op(inp))
                if pred.shape != out.shape or not np.array_equal(pred, out): consistent = False; break
                if not np.array_equal(pred, inp): found_change = True
            except: consistent = False; break
        if consistent and found_change:
            return [np.maximum(ti, op(ti)) for ti in solver.test_in]
    return None
