import numpy as np
from typing import List, Optional

def solve_row_wise_periodic_shift(solver) -> Optional[List[np.ndarray]]:
    consistent_all = True; found_any = False
    for inp, out in solver.pairs:
        h, w = inp.shape
        in_colors = set(np.unique(inp)); out_colors = set(np.unique(out))
        target_colors = list(in_colors - out_colors)
        if len(target_colors) != 1: consistent_all = False; break
        target = target_colors[0]
        
        row_patterns = [] # List of (period, pattern_tuple)
        for r in range(h):
            row = inp[r, :]
            p_pix = [int(c) for c in row if c != target]
            if p_pix:
                p = next((pi for pi in range(1, len(p_pix)+1) if all(p_pix[i] == p_pix[i % pi] for i in range(len(p_pix)))), None)
                if p: row_patterns.append((r, p, tuple(p_pix[:p])))
        
        if not row_patterns: consistent_all = False; break
        
        # Find row-wise periodicity of patterns
        # Map r -> (p, pat)
        r_to_pat = {r: (p, pat) for r, p, pat in row_patterns}
        
        # Find period PR such that Row[r] pattern == Row[r % PR] pattern
        PR = None
        for pr in range(1, h + 1):
            possible = True
            for r in r_to_pat:
                if (r % pr) in r_to_pat:
                    if r_to_pat[r] != r_to_pat[r % pr]:
                        possible = False; break
                else:
                    # We don't know the pattern at r % pr yet, but it must be consistent
                    pass
            if possible:
                # Check if we have enough info to define all rows 0..pr-1
                if all((r % pr) in r_to_pat for r in range(h)):
                    PR = pr; break
                # Actually, even if we don't have all, we can try
                if all((r % pr) in r_to_pat for r in r_to_pat):
                    PR = pr; break
        
        if PR is None: consistent_all = False; break
        
        pred = np.zeros_like(out)
        for r in range(h):
            p, pat = r_to_pat[r % PR]
            # In input at row r, color at col c is pat[(c + offset) % p]
            # But wait, we need the offset for each row!
            # Let's find offset_r for each row in the period
            offset_r = None
            # Find an example row in the input that matches this phase
            example_r = next((er for er in r_to_pat if er % PR == r % PR), None)
            if example_r is not None:
                row = inp[example_r, :]
                offset_r = next((si for si in range(p) if all(row[c]==target or row[c]==pat[(c+si)%p] for c in range(w))), None)
            
            if offset_r is None: consistent_all = False; break
            for c in range(w):
                pred[r, c] = pat[(c + offset_r + 1) % p]
        
        if not np.array_equal(pred, out):
            consistent_all = False; break
        found_any = True
        
    if consistent_all and found_any:
        results = []
        for ti in solver.test_in:
            h, w = ti.shape
            best_res = None
            for target in np.unique(ti):
                row_patterns = []
                for r in range(h):
                    row = ti[r, :]; p_pix = [int(c) for c in row if c != target]
                    if p_pix:
                        p = next((pi for pi in range(1, len(p_pix)+1) if all(p_pix[i] == p_pix[i % pi] for i in range(len(p_pix)))), None)
                        if p: row_patterns.append((r, p, tuple(p_pix[:p])))
                if not row_patterns: continue
                r_to_pat = {r: (p, pat) for r, p, pat in row_patterns}
                PR = next((pr for pr in range(1, h+1) if all(r_to_pat[r]==r_to_pat[r%pr] for r in r_to_pat if (r%pr) in r_to_pat)), None)
                if PR is None: continue
                
                res = np.zeros_like(ti); possible = True
                for r in range(h):
                    if (r % PR) not in r_to_pat: possible = False; break
                    p, pat = r_to_pat[r % PR]
                    example_r = next((er for er in r_to_pat if er % PR == r % PR), None)
                    row = ti[example_r, :]
                    off = next((si for si in range(p) if all(row[c]==target or row[c]==pat[(c+si)%p] for c in range(w))), None)
                    if off is None: possible = False; break
                    for c in range(w): res[r, c] = pat[(c + off + 1) % p]
                if possible: best_res = res; break
            results.append(best_res if best_res is not None else ti.copy())
        return results
    return None
