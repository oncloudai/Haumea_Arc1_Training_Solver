import numpy as np

def solve_extract_3x3_structure_from_divisible_bbox(solver):
    results = []
    
    for ti in solver.test_in:
        ti = np.array(ti)
        unique_colors = np.unique(ti)
        unique_colors = unique_colors[unique_colors != 0]
        color_counts = {c: np.sum(ti == c) for c in unique_colors}
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_colors) == 0: return None
        
        # Try colors to see which one gives a 3x3 divisible bbox
        possible_S = []
        for color, count in sorted_colors:
            coords = np.argwhere(ti == color)
            if len(coords) == 0: continue
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            h, w = r_max - r_min + 1, c_max - c_min + 1
            if h >= 3 and w >= 3 and h % 3 == 0 and w % 3 == 0:
                possible_S.append(color)
                
        if not possible_S:
            S = sorted_colors[0][0]
        else:
            S = possible_S[0]
            
        other_colors = [c for c, count in sorted_colors if c != S]
        N = other_colors[0] if other_colors else S
        
        coords = np.argwhere(ti == S)
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        h, w = r_max - r_min + 1, c_max - c_min + 1
        
        bh, bw = h // 3, w // 3
        out = np.zeros((3, 3), dtype=int)
        for br in range(3):
            for bc in range(3):
                rs, re = r_min + br*bh, r_min + (br+1)*bh
                cs, ce = c_min + bc*bw, c_min + (bc+1)*bw
                sub = ti[rs:re, cs:ce]
                if np.any(sub == S):
                    out[br, bc] = N
        results.append(out)
        
    # Verification
    for inp, outp in solver.pairs:
        inp = np.array(inp)
        outp = np.array(outp)
        if outp.shape != (3, 3): return None
        
        unique_colors = np.unique(inp)
        unique_colors = unique_colors[unique_colors != 0]
        color_counts = {c: np.sum(inp == c) for c in unique_colors}
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
        if not sorted_colors: return None
        
        possible_S = []
        for color, count in sorted_colors:
            coords = np.argwhere(inp == color)
            if len(coords) == 0: continue
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            h, w = r_max - r_min + 1, c_max - c_min + 1
            if h >= 3 and w >= 3 and h % 3 == 0 and w % 3 == 0:
                possible_S.append(color)
        
        if not possible_S: S = sorted_colors[0][0]
        else: S = possible_S[0]
        
        other_colors = [c for c, count in sorted_colors if c != S]
        N = other_colors[0] if other_colors else S
        
        coords = np.argwhere(inp == S)
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        h, w = r_max - r_min + 1, c_max - c_min + 1
        
        bh, bw = h // 3, w // 3
        check_out = np.zeros((3, 3), dtype=int)
        for br in range(3):
            for bc in range(3):
                rs, re = r_min + br*bh, r_min + (br+1)*bh
                cs, ce = c_min + bc*bw, c_min + (bc+1)*bw
                sub = inp[rs:re, cs:ce]
                if np.any(sub == S):
                    check_out[br, bc] = N
        
        if not np.array_equal(check_out, outp):
            return None
            
    return results
