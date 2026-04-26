
import numpy as np

def solve_connect_diagonal_pairs(solver):
    def apply_logic(inp):
        inp = np.array(inp)
        h, w = inp.shape
        outp = inp.copy()
        
        for color in range(1, 10):
            coords = np.argwhere(inp == color)
            if len(coords) == 2:
                r1, c1 = coords[0]
                r2, c2 = coords[1]
                if abs(r2 - r1) == abs(c2 - c1) and abs(r2 - r1) > 0:
                    dr = 1 if r2 > r1 else -1
                    dc = 1 if c2 > c1 else -1
                    steps = abs(r2 - r1)
                    for i in range(1, steps):
                        outp[r1 + i*dr, c1 + i*dc] = color
                else:
                    # Maybe not diagonal?
                    return None
            elif len(coords) > 2 or len(coords) == 1:
                # Not a pair
                pass
        return outp

    results = []
    for inp, outp in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, outp):
            return None
            
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
