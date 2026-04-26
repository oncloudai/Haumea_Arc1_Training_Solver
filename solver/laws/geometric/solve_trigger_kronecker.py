import numpy as np
from typing import List, Optional
from collections import Counter

def solve_trigger_kronecker(solver) -> Optional[List[np.ndarray]]:
    h_in, w_in = solver.train_in[0].shape
    h_out, w_out = solver.train_out[0].shape
    if h_out != h_in * h_in or w_out != w_in * w_in: return None
    
    # Try different rules for selecting trigger color
    rules = []
    # Rule 1-9: Fixed color C
    for c in range(1, 10):
        rules.append(lambda inp, c=c: c)
    # Rule 10: Most frequent color (excluding 0)
    def most_freq(inp):
        counts = Counter(inp.flatten())
        if 0 in counts: del counts[0]
        return counts.most_common(1)[0][0] if counts else None
    rules.append(most_freq)
    # Rule 11: Least frequent color (excluding 0)
    def least_freq(inp):
        counts = Counter(inp.flatten())
        if 0 in counts: del counts[0]
        return counts.most_common()[-1][0] if counts else None
    rules.append(least_freq)
    
    for rule in rules:
        consistent = True; found_any = False
        for inp, out in solver.pairs:
            tc = rule(inp)
            if tc is None: consistent = False; break
            
            pred = np.zeros_like(out)
            trigger_indices = np.argwhere(inp == tc)
            if len(trigger_indices) == 0: consistent = False; break
            
            for r, c in trigger_indices:
                pred[r*h_in:(r+1)*h_in, c*w_in:(c+1)*w_in] = inp
                found_any = True
            
            if not np.array_equal(pred, out):
                consistent = False; break
        
        if consistent and found_any:
            results = []
            for ti in solver.test_in:
                tc = rule(ti)
                if tc is None: results.append(ti.copy()); continue
                h, w = ti.shape
                res = np.zeros((h*h, w*w), dtype=int)
                trigger_indices = np.argwhere(ti == tc)
                for r, c in trigger_indices:
                    res[r*h:(r+1)*h, c*w:(c+1)*w] = ti
                results.append(res)
            return results
    return None
