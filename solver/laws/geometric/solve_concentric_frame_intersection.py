import numpy as np

def solve_concentric_frame_intersection(solver):
    results = []
    
    for ti in solver.test_in:
        ti = np.array(ti)
        dots = np.argwhere(ti != 0)
        if len(dots) < 2: return None
        
        # Sort dots by coordinate
        dots = dots[np.lexsort((dots[:, 1], dots[:, 0]))]
        
        # Determine strides
        strides = [1] * len(dots)
        if len(dots) == 3:
            dr = abs(dots[1, 0] - dots[0, 0])
            dc = abs(dots[1, 1] - dots[0, 1])
            strides[1] = max(dr, dc)
        elif len(dots) == 2:
            # Just union? Or both 1?
            # Let's assume for 2 dots they both have some stride
            pass
            
        color = ti[dots[0,0], dots[0,1]]
        h, w = ti.shape
        out = np.zeros_like(ti)
        
        for r in range(h):
            for c in range(w):
                on_all = True
                for i, (rd, cd) in enumerate(dots):
                    d = max(abs(r-rd), abs(c-cd))
                    if d % strides[i] != 0:
                        on_all = False
                        break
                if on_all:
                    out[r, c] = color
        results.append(out)
        
    # Verification
    for inp, outp in solver.pairs:
        inp = np.array(inp)
        outp = np.array(outp)
        dots = np.argwhere(inp != 0)
        if len(dots) < 2: return None
        dots = dots[np.lexsort((dots[:, 1], dots[:, 0]))]
        
        strides = [1] * len(dots)
        if len(dots) == 3:
            dr = abs(dots[1, 0] - dots[0, 0])
            dc = abs(dots[1, 1] - dots[0, 1])
            strides[1] = max(dr, dc)
            
        color = inp[dots[0,0], dots[0,1]]
        h, w = inp.shape
        check_out = np.zeros_like(inp)
        for r in range(h):
            for c in range(w):
                on_all = True
                for i, (rd, cd) in enumerate(dots):
                    d = max(abs(r-rd), abs(c-cd))
                    if d % strides[i] != 0:
                        on_all = False
                        break
                if on_all:
                    check_out[r, c] = color
                    
        if not np.array_equal(check_out, outp):
            return None
            
    return results
