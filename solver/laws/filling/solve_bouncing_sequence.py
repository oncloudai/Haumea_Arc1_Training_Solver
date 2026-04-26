import numpy as np
from typing import List, Optional

def solve_bouncing_sequence(solver) -> Optional[List[np.ndarray]]:
    for color in range(1, 10):
        consistent = True; found_any = False
        for inp, out in solver.pairs:
            coords = np.argwhere(inp == color)
            if len(coords) != 1: consistent = False; break
            
            r0, c0 = coords[0]
            H, W = inp.shape
            
            # Sequence: 0, 1, ..., W-1, W-2, ..., 1
            seq = list(range(W)) + list(range(W-2, 0, -1))
            if not seq: seq = [0]
            
            # Find all indices of c0 in seq
            indices = [i for i, x in enumerate(seq) if x == c0]
            
            best_pred = None
            for start_idx in indices:
                pred = inp.copy()
                # Check both directions? 
                # Actually, let's try starting from start_idx and moving forward in seq
                for r in range(r0, -1, -1):
                    idx = (start_idx + (r0 - r)) % len(seq)
                    pred[r, seq[idx]] = color
                
                if np.array_equal(pred, out):
                    best_pred = pred; break
                
                # Try backward in seq
                pred = inp.copy()
                for r in range(r0, -1, -1):
                    idx = (start_idx - (r0 - r)) % len(seq)
                    pred[r, seq[idx]] = color
                if np.array_equal(pred, out):
                    best_pred = pred; break
            
            if best_pred is None: consistent = False; break
            found_any = True
            
        if consistent and found_any:
            results = []
            for ti in solver.test_in:
                coords = np.argwhere(ti == color)
                if len(coords) != 1: results.append(ti.copy()); continue
                r0, c0 = coords[0]; H, W = ti.shape
                seq = list(range(W)) + list(range(W-2, 0, -1))
                if not seq: seq = [0]
                indices = [i for i, x in enumerate(seq) if x == c0]
                
                # We need to know which start_idx and which direction worked.
                # Since we don't store it, we can try to find it again using first training pair.
                inp0, out0 = solver.pairs[0]
                r0_0, c0_0 = np.argwhere(inp0 == color)[0]
                seq_0 = list(range(inp0.shape[1])) + list(range(inp0.shape[1]-2, 0, -1))
                if not seq_0: seq_0 = [0]
                indices_0 = [i for i, x in enumerate(seq_0) if x == c0_0]
                
                found_dir = None
                for si in indices_0:
                    p = inp0.copy()
                    for r in range(r0_0, -1, -1): p[r, seq_0[(si + (r0_0-r)) % len(seq_0)]] = color
                    if np.array_equal(p, out0): found_dir = ('fwd', si); break
                    p = inp0.copy()
                    for r in range(r0_0, -1, -1): p[r, seq_0[(si - (r0_0-r)) % len(seq_0)]] = color
                    if np.array_equal(p, out0): found_dir = ('bwd', si); break
                
                if found_dir:
                    direction, _ = found_dir
                    # Note: 'si' is for training. For test, we use 'indices[0]' or similar.
                    # Wait, if there are multiple indices, we might need more logic.
                    # But usually c0=0, so index is 0.
                    res = ti.copy()
                    si_test = indices[0] # Just take the first one for now
                    for r in range(r0, -1, -1):
                        if direction == 'fwd': idx = (si_test + (r0 - r)) % len(seq)
                        else: idx = (si_test - (r0 - r)) % len(seq)
                        res[r, seq[idx]] = color
                    results.append(res)
                else:
                    results.append(ti.copy())
            return results
    return None
