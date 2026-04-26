
import numpy as np
from solver.utils import get_homogeneous_blobs

def solve_count_blobs_as_grid(solver):
    # This law counts blobs of color C and size S, then produces a 1xN grid with that many color C pixels.
    for color in range(1, 10):
        for size in [1, 2, 4, 9]: # Common blob sizes
            consistent = True
            for inp, outp in solver.pairs:
                blobs = get_homogeneous_blobs(inp, background=0)
                count = 0
                for b in blobs:
                    if b['color'] == color and b['size'] == size:
                        # For size 4, verify 2x2? Let's just use size for now.
                        count += 1
                
                # Check if outp matches
                oh, ow = outp.shape
                if oh != 1: consistent = False; break
                pred = np.zeros_like(outp)
                for i in range(min(count, ow)):
                    pred[0, i] = color
                if not np.array_equal(pred, outp):
                    consistent = False; break
            
            if consistent:
                results = []
                for ti in solver.test_in:
                    blobs = get_homogeneous_blobs(ti, background=0)
                    count = 0
                    for b in blobs:
                        if b['color'] == color and b['size'] == size:
                            count += 1
                    oh, ow = solver.pairs[0][1].shape # Output shape from train
                    pred_ti = np.zeros((oh, ow), dtype=int)
                    for i in range(min(count, ow)):
                        pred_ti[0, i] = color
                    results.append(pred_ti)
                return results
    return None
