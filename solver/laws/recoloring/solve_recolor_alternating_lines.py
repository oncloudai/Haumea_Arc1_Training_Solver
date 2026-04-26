
import numpy as np
from typing import List, Optional

def solve_recolor_alternating_lines(solver) -> Optional[List[np.ndarray]]:
    """
    Identifies a source color and a target color, and recolors pixels 
    of the source color based on row or column parity.
    """
    consistent = True; found_any = False
    
    # Try all color pairs
    best_rule = None
    for src in range(1, 10):
        for dst in range(1, 10):
            if src == dst: continue
            for axis in ['row', 'col']:
                for offset in [0, 1]:
                    rule_consistent = True; rule_found_change = False
                    for inp, out in solver.pairs:
                        pred = inp.copy()
                        for r in range(inp.shape[0]):
                            for c in range(inp.shape[1]):
                                if inp[r, c] == src:
                                    val = r if axis == 'row' else c
                                    if (val + offset) % 2 == 1:
                                        pred[r, c] = dst
                                        rule_found_change = True
                        if not np.array_equal(pred, out):
                            rule_consistent = False; break
                    
                    if rule_consistent and rule_found_change:
                        best_rule = (src, dst, axis, offset)
                        break
                if best_rule: break
            if best_rule: break
        if best_rule: break
        
    if best_rule:
        src, dst, axis, offset = best_rule
        results = []
        for ti in solver.test_in:
            res = ti.copy()
            for r in range(ti.shape[0]):
                for c in range(ti.shape[1]):
                    if ti[r, c] == src:
                        val = r if axis == 'row' else c
                        if (val + offset) % 2 == 1:
                            res[r, c] = dst
            results.append(res)
        return results
    return None
