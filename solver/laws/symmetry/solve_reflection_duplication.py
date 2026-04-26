import numpy as np
from typing import List, Optional

def solve_reflection_duplication(solver) -> Optional[List[np.ndarray]]:
    # Try horizontal flip then vertical flip
    for mode in ['h', 'v']:
        consistent = True
        for inp, out in solver.pairs:
            if mode == 'h':
                pred = np.hstack([inp, np.fliplr(inp)])
            else:
                pred = np.vstack([inp, np.flipud(inp)])
            
            if out.shape != pred.shape or not np.array_equal(pred, out):
                consistent = False; break
        
        if consistent:
            results = []
            for ti in solver.test_in:
                if mode == 'h':
                    res = np.hstack([ti, np.fliplr(ti)])
                else:
                    res = np.vstack([ti, np.flipud(ti)])
                results.append(res)
            return results
    return None
