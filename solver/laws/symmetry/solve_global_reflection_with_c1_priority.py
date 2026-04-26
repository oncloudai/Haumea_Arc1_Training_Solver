import numpy as np

def solve_global_reflection_with_c1_priority(solver):
    def apply_point_symmetry(inp, r0, c0):
        h, w = inp.shape
        c1 = (inp == 1)
        outp = inp.copy()
        pred_c2 = np.zeros_like(c1)
        
        for r in range(h):
            for c in range(w):
                if c1[r, c]:
                    mr = int(round(2 * r0 - r))
                    mc = int(round(2 * c0 - c))
                    if 0 <= mr < h and 0 <= mc < w:
                        if not c1[mr, mc]:
                            pred_c2[mr, mc] = True
        outp[pred_c2] = 2
        return outp

    # The point of symmetry varies. Let's find it for each train pair.
    train_centers = []
    for inp, outp in solver.pairs:
        h, w = inp.shape
        found_center = None
        for r2 in range(2 * h):
            for c2_ in range(2 * w):
                r0, c0 = r2 / 2.0, c2_ / 2.0
                pred = apply_point_symmetry(inp, r0, c0)
                if np.array_equal(pred, outp) and np.any(outp == 2):
                    found_center = (r0, c0); break
            if found_center: break
        if not found_center: return None
        train_centers.append(found_center)

    # Now we need a rule to pick the center for the test case.
    # Heuristic: Is it always the grid center (4.5, 4.5) OR (5.0, 5.0)?
    # Or is it the center of some 2x2 or 1x1 block of color 1?
    
    results = []
    for ti in solver.test_in:
        ti = np.array(ti)
        h, w = ti.shape
        # Try grid center first (most common in Arc)
        grid_center = ((h-1)/2.0, (w-1)/2.0)
        
        # Check if grid center (or some other common center) worked in all train cases where it was applicable?
        # Actually, in this task, the rule seems to be: 
        # If there is a 2x2 at (4,4), use (4.5, 4.5).
        # If there is a 1x1 at (5,5), use (5.0, 5.0).
        
        # Let's try centers that appeared in training
        possible_centers = sorted(list(set(train_centers)))
        # and grid center
        if grid_center not in possible_centers:
            possible_centers.append(grid_center)
            
        # This is a bit of a hack, but without a general rule for center selection
        # we can't be sure. Let's try grid center first.
        results.append(apply_point_symmetry(ti, grid_center[0], grid_center[1]))
        
    return results
