import numpy as np
from typing import List, Optional

def solve_periodic_shift_completion(solver) -> Optional[List[np.ndarray]]:
    for P in range(2, 5): # Period
        for A in [-1, 0, 1]:
            for B in [-1, 0, 1]:
                if A == 0 and B == 0: continue
                
                consistent = True; found_any = False; global_mapping = None
                for inp, out in solver.pairs:
                    h, w = out.shape
                    # Find mapping for this pair
                    mapping = {}
                    for r in range(h):
                        for c in range(w):
                            v = (A*r + B*c) % P
                            if v in mapping:
                                if mapping[v] != out[r, c]: consistent = False; break
                            else: mapping[v] = out[r, c]
                        if not consistent: break
                    if not consistent: break
                    
                    # Verify the "shift by 1" or "inversion" rule
                    # Find input colors at these phases
                    inp_mapping = {}
                    for r in range(h):
                        for c in range(w):
                            if inp[r, c] != 0: # Assuming 0 or target color is not part of pattern
                                # Wait, how to know target color? It's the one that disappears.
                                pass
                    
                    # Instead of complex rule, let's just see if the mapping is consistent 
                    # across training pairs if we relative it to something.
                    # But the colors change. So we need a rule to find 'mapping'.
                    
                    # Let's try: mapping[v] = most_frequent_color_at_phase_v_shifted?
                    # Actually, the user's hint "Out(0,0) != In(0,0)" is key.
                    # In Train: Out(r,c) = In_Pattern( (r+c+1)%2 )
                    # In Test:  Out(r,c) = In_Pattern( (r+c+1)%3 )
                    
                    # Let's find the pattern colors in order of appearance.
                    # Pair 0: 6, 7. Target 3.
                    # Phase 0 (0+0)%2: In=6, Out=7.
                    # Phase 1 (0+1)%2: In=7, Out=6.
                    # This is exactly mapping Phase V to color at Phase (V+1)%P.
                    
                    # Let's test this "Phase V -> In_Color_at_Phase_(V+1)%P" rule.
                    pattern_colors = []
                    for v in range(P):
                        # Find color at phase v in input
                        # (Need to find a pixel where (A*r + B*c)%P == v and inp[r,c] is not target)
                        pass
                    
                # To simplify, let's just find the best mapping for each pair and see if 
                # they follow a consistent shift rule.
                
    # Re-trying a simpler approach:
    # 1. Identify target color (the one that is in IN but not OUT).
    # 2. Identify pattern period P and direction (A, B).
    # 3. For each phase v in [0, P-1], find the color C_v in the input pattern.
    # 4. The output at (r, c) is C_{ (A*r + B*c + 1) % P }.
    
    for P in range(2, 4):
        for A in [-1, 0, 1]:
            for B in [-1, 0, 1]:
                if A == 0 and B == 0: continue
                consistent_all = True; found_any = False
                for inp, out in solver.pairs:
                    h, w = inp.shape
                    # Target color is the one in inp but not in out
                    in_colors = set(np.unique(inp))
                    out_colors = set(np.unique(out))
                    target_color = list(in_colors - out_colors)
                    if len(target_color) != 1: consistent_all = False; break
                    target_color = target_color[0]
                    
                    # Map phase -> input color
                    phase_to_in_color = {}
                    for r in range(h):
                        for c in range(w):
                            if inp[r, c] != target_color:
                                v = (A*r + B*c) % P
                                if v in phase_to_in_color:
                                    if phase_to_in_color[v] != inp[r, c]:
                                        # Not a simple periodic pattern
                                        consistent_all = False; break
                                else:
                                    phase_to_in_color[v] = inp[r, c]
                        if not consistent_all: break
                    if not consistent_all or len(phase_to_in_color) < P:
                        consistent_all = False; break
                    
                    # Predict out[r, c] = phase_to_in_color[ (v + 1) % P ]
                    pred = np.zeros_like(out)
                    for r in range(h):
                        for c in range(w):
                            v = (A*r + B*c) % P
                            pred[r, c] = phase_to_in_color[(v + 1) % P]
                    
                    if not np.array_equal(pred, out):
                        consistent_all = False; break
                    found_any = True
                    
                if consistent_all and found_any:
                    results = []
                    for ti in solver.test_in:
                        h, w = ti.shape
                        in_colors = set(np.unique(ti))
                        # We don't have 'out' for test, so how to find target_color?
                        # It's the one that forms large solid blocks?
                        # Or the most frequent? In all training, it was the one 
                        # that didn't fit the pattern.
                        
                        # Let's find the color that DOES NOT fit the (A,B,P) pattern.
                        potential_targets = []
                        for c in in_colors:
                            # If we assume c is the target, is the rest periodic?
                            is_periodic = True; mapping = {}
                            for r in range(h):
                                for c2 in range(w):
                                    if ti[r, c2] != c:
                                        v = (A*r + B*c2) % P
                                        if v in mapping:
                                            if mapping[v] != ti[r, c2]: is_periodic = False; break
                                        else: mapping[v] = ti[r, c2]
                                if not is_periodic: break
                            if is_periodic and len(mapping) == P:
                                potential_targets.append((c, mapping))
                        
                        if not potential_targets:
                            results.append(ti.copy()); continue
                        
                        # Assume first potential target
                        target, phase_to_in_color = potential_targets[0]
                        res = np.zeros_like(ti)
                        for r in range(h):
                            for c2 in range(w):
                                v = (A*r + B*c2) % P
                                res[r, c2] = phase_to_in_color[(v + 1) % P]
                        results.append(res)
                    return results
    return None
