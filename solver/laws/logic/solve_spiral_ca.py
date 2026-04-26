import numpy as np
from typing import List, Optional
from collections import Counter

def solve_one(inp, color=3):
    h, w = inp.shape
    out = np.zeros((h, w), dtype=int)
    r, c = 0, 0
    dr, dc = 0, 1 # Start moving right
    
    # Leg 1
    L = w
    for i in range(L):
        out[r, c] = color
        if i < L - 1:
            r += dr
            c += dc
    
    leg_idx = 2
    while True:
        # Turn right
        dr, dc = dc, -dr
        
        # Formula for leg lengths:
        # L2, L3 = N-1
        # L4, L5 = N-3
        # L6, L7 = N-5
        k = leg_idx // 2
        L = h - (2*k - 1)
        
        if L <= 0:
            break
            
        for _ in range(L):
            r += dr
            c += dc
            out[r, c] = color
            
        leg_idx += 1
    return out

def solve_spiral_ca(solver) -> Optional[List[np.ndarray]]:
    # Find the spiral color from the first training pair
    if not solver.pairs: return None
    inp_0, out_0 = solver.pairs[0]
    diff = (out_0 != inp_0)
    if not np.any(diff):
        return None
    # Get the most common color that changed
    changed_colors = out_0[diff]
    counts = Counter(changed_colors.flatten())
    if 0 in counts: del counts[0]
    if not counts: return None
    color = int(counts.most_common(1)[0][0])
    
    # Verify on all training pairs
    for inp, out in solver.pairs:
        pred = solve_one(inp, color)
        if not np.array_equal(pred, out):
            return None
            
    # Apply to test
    return [solve_one(ti, color) for ti in solver.test_in]
