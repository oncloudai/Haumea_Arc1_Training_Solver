
import numpy as np
from typing import List, Optional

def solve_dominant_sequence_extraction(solver) -> Optional[List[np.ndarray]]:
    """
    Finds if all rows or all columns share the same sequence of distinct non-zero colors.
    If so, extracts that sequence as a 1xN or Nx1 grid.
    """
    def get_distinct_sequence(seq):
        if len(seq) == 0: return []
        res = [seq[0]]
        for x in seq[1:]:
            if x != res[-1]:
                res.append(x)
        return [x for x in res if x != 0]

    def apply_logic(inp):
        inp = np.array(inp)
        h, w = inp.shape
        
        # Check rows
        row_seqs = []
        for r in range(h):
            s = get_distinct_sequence(inp[r, :].tolist())
            if s:
                if not row_seqs or s != row_seqs[0]:
                    row_seqs.append(s)
        
        if len(row_seqs) == 1:
            return np.array([row_seqs[0]])
            
        # Check columns
        col_seqs = []
        for c in range(w):
            s = get_distinct_sequence(inp[:, c].tolist())
            if s:
                if not col_seqs or s != col_seqs[0]:
                    col_seqs.append(s)
                    
        if len(col_seqs) == 1:
            return np.array([[c] for c in col_seqs[0]])
            
        return None

    for inp, out in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, out):
            return None
            
    return [apply_logic(ti) for ti in solver.test_in]
