
import numpy as np
from solver.utils import get_blobs

def solve_align_objects_to_anchor(solver):
    # This law aligns all objects vertically to the same rows as an anchor color.
    # First, find which color acts as the anchor (stays in its rows, or everyone moves to its rows)
    
    for anchor_color in range(1, 10):
        consistent = True
        results_train = []
        for inp, outp in solver.pairs:
            inp = np.array(inp)
            outp = np.array(outp)
            blobs_in = get_blobs(inp, background=0)
            if not blobs_in: consistent = False; break
            
            anchor_blobs = [b for b in blobs_in if b['color'] == anchor_color]
            if not anchor_blobs: consistent = False; break
            
            # Use the first blob of anchor color as the vertical reference
            ref_r_min = anchor_blobs[0]['top_left'][0]
            
            pred = np.zeros_like(outp)
            for b in blobs_in:
                rel_coords = b['coords'] - b['top_left']
                new_tl_r = ref_r_min
                new_tl_c = b['top_left'][1]
                for dr, dc in rel_coords:
                    tr, tc = new_tl_r + dr, new_tl_c + dc
                    if 0 <= tr < pred.shape[0] and 0 <= tc < pred.shape[1]:
                        pred[tr, tc] = b['color']
            
            if not np.array_equal(pred, outp):
                consistent = False; break
                
        if consistent:
            results = []
            for ti in solver.test_in:
                ti = np.array(ti)
                blobs_ti = get_blobs(ti, background=0)
                anchor_blobs = [b for b in blobs_ti if b['color'] == anchor_color]
                if not anchor_blobs: return None # Anchor missing in test
                
                ref_r_min = anchor_blobs[0]['top_left'][0]
                pred_ti = np.zeros_like(ti)
                for b in blobs_ti:
                    rel_coords = b['coords'] - b['top_left']
                    new_tl_r = ref_r_min
                    new_tl_c = b['top_left'][1]
                    for dr, dc in rel_coords:
                        tr, tc = new_tl_r + dr, new_tl_c + dc
                        if 0 <= tr < pred_ti.shape[0] and 0 <= tc < pred_ti.shape[1]:
                            pred_ti[tr, tc] = b['color']
                results.append(pred_ti)
            return results
            
    return None
