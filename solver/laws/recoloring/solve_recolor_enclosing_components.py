import numpy as np
from typing import List, Optional
from scipy.ndimage import binary_fill_holes
from solver.utils import get_blobs

def solve_recolor_enclosing_components(solver) -> Optional[List[np.ndarray]]:
    for bg in range(10):
        # Identify possible colors to recolor
        for c_in in range(10):
            if c_in == bg: continue
            for c_out in range(10):
                if c_out == bg or c_out == c_in: continue
                
                consistent = True
                found_any = False
                for inp, out in solver.pairs:
                    if inp.shape != out.shape: consistent = False; break
                    
                    # We need to try both connectivities
                    solved_with_conn = None
                    for conn in [8, 4]:
                        pred = inp.copy()
                        blobs = get_blobs(inp, bg, connectivity=conn)
                        has_change = False
                        for b in blobs:
                            if b['color'] == c_in:
                                # Check if blob encloses any holes
                                min_r, min_c = b['coords'].min(axis=0)
                                max_r, max_c = b['coords'].max(axis=0)
                                # Create a local mask with padding
                                h, w = max_r - min_r + 1, max_c - min_c + 1
                                mask = np.zeros((h + 2, w + 2), dtype=bool)
                                for r, c in b['coords']:
                                    mask[r - min_r + 1, c - min_c + 1] = True
                                
                                filled = binary_fill_holes(mask)
                                if not np.array_equal(filled, mask):
                                    # Blob has holes
                                    for r, c in b['coords']:
                                        pred[r, c] = c_out
                                    has_change = True
                        
                        if np.array_equal(pred, out):
                            solved_with_conn = conn
                            if has_change: found_any = True
                            break
                    
                    if solved_with_conn is None:
                        consistent = False; break
                
                if consistent and found_any:
                    # Verify consistency of solved_with_conn across all pairs
                    # But actually it might vary? Better to stick to one.
                    # Let's find the conn that worked for ALL.
                    for conn in [8, 4]:
                        all_match = True
                        for inp, out in solver.pairs:
                            pred = inp.copy()
                            blobs = get_blobs(inp, bg, connectivity=conn)
                            for b in blobs:
                                if b['color'] == c_in:
                                    min_r, min_c = b['coords'].min(axis=0)
                                    max_r, max_c = b['coords'].max(axis=0)
                                    h, w = max_r - min_r + 1, max_c - min_c + 1
                                    mask = np.zeros((h + 2, w + 2), dtype=bool)
                                    for r, c in b['coords']:
                                        mask[r - min_r + 1, c - min_c + 1] = True
                                    if not np.array_equal(binary_fill_holes(mask), mask):
                                        for r, c in b['coords']: pred[r, c] = c_out
                            if not np.array_equal(pred, out):
                                all_match = False; break
                        
                        if all_match:
                            results = []
                            for ti in solver.test_in:
                                res = ti.copy()
                                t_blobs = get_blobs(ti, bg, connectivity=conn)
                                for b in t_blobs:
                                    if b['color'] == c_in:
                                        min_r, min_c = b['coords'].min(axis=0)
                                        max_r, max_c = b['coords'].max(axis=0)
                                        h, w = max_r - min_r + 1, max_c - min_c + 1
                                        mask = np.zeros((h + 2, w + 2), dtype=bool)
                                        for r, c in b['coords']:
                                            mask[r - min_r + 1, c - min_c + 1] = True
                                        if not np.array_equal(binary_fill_holes(mask), mask):
                                            for r, c in b['coords']: res[r, c] = c_out
                                results.append(res)
                            return results
    return None
