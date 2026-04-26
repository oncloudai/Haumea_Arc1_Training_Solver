import numpy as np

def solve_connect_same_color_orthogonal_pairs(solver):
    def apply_logic(inp, anchor_color, filler_color):
        inp = np.array(inp)
        h, w = inp.shape
        outp = inp.copy()
        coords = np.argwhere(inp == anchor_color)
        if len(coords) < 2: return outp
        
        used = set()
        coords_list = sorted([tuple(c) for c in coords])
        
        for i in range(len(coords_list)):
            if i in used: continue
            r1, c1 = coords_list[i]
            best_j = -1
            min_dist = float('inf')
            
            for j in range(len(coords_list)):
                if i == j or j in used: continue
                r2, c2 = coords_list[j]
                if r1 == r2:
                    dist = abs(c2 - c1)
                    if dist < min_dist:
                        min_dist = dist
                        best_j = j
            
            if best_j != -1:
                r2, c2 = coords_list[best_j]
                outp[r1, min(c1, c2)+1:max(c1, c2)] = filler_color
                used.add(i); used.add(best_j)
                continue
                
            min_dist = float('inf')
            for j in range(len(coords_list)):
                if i == j or j in used: continue
                r2, c2 = coords_list[j]
                if c1 == c2:
                    dist = abs(r2 - r1)
                    if dist < min_dist:
                        min_dist = dist
                        best_j = j
            
            if best_j != -1:
                r2, c2 = coords_list[best_j]
                outp[min(r1, r2)+1:max(r1, r2), c1] = filler_color
                used.add(i); used.add(best_j)
                
        return outp

    for anchor in range(1, 10):
        for filler in range(1, 10):
            if anchor == filler: continue
            consistent = True
            for inp, outp in solver.pairs:
                pred = apply_logic(inp, anchor, filler)
                if not np.array_equal(pred, outp):
                    consistent = False; break
            
            if consistent:
                results = []
                for ti in solver.test_in:
                    results.append(apply_logic(ti, anchor, filler))
                return results
    return None
