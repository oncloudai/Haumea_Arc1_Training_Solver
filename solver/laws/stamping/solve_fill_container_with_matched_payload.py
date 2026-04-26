import numpy as np
from solver.utils import get_blobs, get_enclosed_holes
from scipy.ndimage import label

def solve_fill_container_with_matched_payload(solver):
    def apply_logic(inp):
        inp = np.array(inp)
        h, w = inp.shape
        
        container_mask = (inp == 5)
        labeled_containers, n_containers = label(container_mask)
        if n_containers == 0: return None
        
        payload_blobs = get_blobs(inp, background=0, connectivity=8)
        # Payloads are not 5 and not 0.
        payload_blobs = [b for b in payload_blobs if b['color'] != 5]
        
        outp = np.zeros_like(inp)
        outp[container_mask] = 5
        
        # We also need to keep pixels that are NOT 0, NOT 5, and NOT part of any payload blob?
        # Actually every non-zero pixel is part of SOME blob.
        
        used_payloads = set()
        for i in range(1, n_containers + 1):
            cont_mask = (labeled_containers == i)
            temp_grid = np.zeros_like(inp)
            temp_grid[cont_mask] = 5
            holes = get_enclosed_holes(temp_grid, 5)
            
            for hole in holes:
                hole_coords = hole['coords']
                hole_min = hole_coords.min(axis=0)
                hole_shape_set = set(tuple(x) for x in (hole_coords - hole_min))
                
                for p_idx, p in enumerate(payload_blobs):
                    if p_idx in used_payloads: continue
                    p_shape_set = set(tuple(x) for x in (p['coords'] - p['top_left']))
                    if hole_shape_set == p_shape_set:
                        for hr, hc in hole_coords: outp[hr, hc] = p['color']
                        used_payloads.add(p_idx); break
        
        # Non-container, non-zero pixels from input
        for p_idx, p in enumerate(payload_blobs):
            if p_idx not in used_payloads:
                for pr, pc in p['coords']:
                    # Only if it was NOT moved into a hole? 
                    # If it was moved, it's in used_payloads.
                    outp[pr, pc] = p['color']
                    
        # Final check: are there any pixels in outp that were 0 in inp but now non-zero?
        # Yes, the ones in holes.
        # Are there any pixels that were non-zero in inp but now 0?
        # Yes, the used payloads.
        
        return outp

    results = []
    for inp, outp in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, outp): return None
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
