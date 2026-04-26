import numpy as np

def solve_crop_between_fences(solver):
    results = []
    
    for ti in solver.test_in:
        ti = np.array(ti)
        
        # Find fences (rows/cols that are mostly one non-zero color)
        fence_rows = []
        for r in range(ti.shape[0]):
            counts = np.bincount(ti[r, :].flatten(), minlength=10)
            for color in range(1, 10):
                if counts[color] >= ti.shape[1] - 4:
                    fence_rows.append((r, color))
                    break
        
        fence_cols = []
        for c in range(ti.shape[1]):
            counts = np.bincount(ti[:, c].flatten(), minlength=10)
            for color in range(1, 10):
                if counts[color] >= ti.shape[0] - 4:
                    fence_cols.append((c, color))
                    break
        
        if len(fence_rows) < 2 or len(fence_cols) < 2:
            return None
            
        fence_rows.sort()
        fence_cols.sort()
        
        r_start, r_start_color = fence_rows[0]
        r_end, r_end_color = fence_rows[-1]
        c_start, c_start_color = fence_cols[0]
        c_end, c_end_color = fence_cols[-1]
        
        out = ti[r_start:r_end+1, c_start:c_end+1].copy()
        h, w = out.shape
        
        fence_colors = {r_start_color, r_end_color, c_start_color, c_end_color}
        
        for color in fence_colors:
            # Vertical projection if color matches a horizontal fence color
            if color == r_start_color or color == r_end_color:
                for c in range(w):
                    rows = np.where(out[:, c] == color)[0]
                    if len(rows) > 0:
                        r_min, r_max = rows.min(), rows.max()
                        if color == r_start_color:
                            out[0:r_max+1, c] = color
                        if color == r_end_color:
                            out[r_min:h, c] = color
            
            # Horizontal projection if color matches a vertical fence color
            if color == c_start_color or color == c_end_color:
                for r in range(h):
                    cols = np.where(out[r, :] == color)[0]
                    if len(cols) > 0:
                        c_min, c_max = cols.min(), cols.max()
                        if color == c_start_color:
                            out[r, 0:c_max+1] = color
                        if color == c_end_color:
                            out[r, c_min:w] = color
                            
        results.append(out)
        
    # Verification
    for inp, outp in solver.pairs:
        inp = np.array(inp)
        outp = np.array(outp)
        
        fence_rows = []
        for r in range(inp.shape[0]):
            counts = np.bincount(inp[r, :].flatten(), minlength=10)
            for color in range(1, 10):
                if counts[color] >= inp.shape[1] - 4:
                    fence_rows.append((r, color))
                    break
        
        fence_cols = []
        for c in range(inp.shape[1]):
            counts = np.bincount(inp[:, c].flatten(), minlength=10)
            for color in range(1, 10):
                if counts[color] >= inp.shape[0] - 4:
                    fence_cols.append((c, color))
                    break
                    
        if len(fence_rows) < 2 or len(fence_cols) < 2:
            return None
            
        fence_rows.sort()
        fence_cols.sort()
        
        r_s, rs_c = fence_rows[0]
        r_e, re_c = fence_rows[-1]
        c_s, cs_c = fence_cols[0]
        c_e, ce_c = fence_cols[-1]
        
        check_out = inp[r_s:r_e+1, c_s:c_e+1].copy()
        h, w = check_out.shape
        f_colors = {rs_c, re_c, cs_c, ce_c}
        
        for color in f_colors:
            if color == rs_c or color == re_c:
                for c in range(w):
                    rows = np.where(check_out[:, c] == color)[0]
                    if len(rows) > 0:
                        r_min, r_max = rows.min(), rows.max()
                        if color == rs_c: check_out[0:r_max+1, c] = color
                        if color == re_c: check_out[r_min:h, c] = color
            if color == cs_c or color == ce_c:
                for r in range(h):
                    cols = np.where(check_out[r, :] == color)[0]
                    if len(cols) > 0:
                        c_min, c_max = cols.min(), cols.max()
                        if color == cs_c: check_out[r, 0:c_max+1] = color
                        if color == ce_c: check_out[r, c_min:w] = color
                        
        if check_out.shape != outp.shape or not np.array_equal(check_out, outp):
            return None

    return results
