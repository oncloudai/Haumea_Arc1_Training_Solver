
import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_one(inp):
    inp = np.array(inp)
    h, w = inp.shape
    # Background is the most common color
    counts = np.bincount(inp.flatten())
    bg = int(counts.argmax())
    out = inp.copy()
    
    mask = (inp != bg)
    labeled, num = label(mask)
    
    blobs = []
    for i in range(1, num + 1):
        coords = np.argwhere(labeled == i)
        color = inp[tuple(coords[0])]
        blobs.append({'color': color, 'coords': coords, 'size': len(coords)})
        
    for i, b1 in enumerate(blobs):
        if b1['size'] != 1:
            continue
        r1, c1 = b1['coords'][0]
        
        for j, b2 in enumerate(blobs):
            if i == j or b1['color'] == b2['color']:
                continue
            
            # Horizontal projection
            same_row = b2['coords'][b2['coords'][:, 0] == r1]
            if len(same_row) > 0:
                dist = same_row[:, 1] - c1
                if np.any(dist > 0):
                    target_c = same_row[dist > 0, 1].min()
                    if np.all(inp[r1, c1+1:target_c] == bg):
                        out[r1, c1+1:target_c] = b1['color']
                if np.any(dist < 0):
                    target_c = same_row[dist < 0, 1].max()
                    if np.all(inp[r1, target_c+1:c1] == bg):
                        out[r1, target_c+1:c1] = b1['color']
                        
            # Vertical projection
            same_col = b2['coords'][b2['coords'][:, 1] == c1]
            if len(same_col) > 0:
                dist = same_col[:, 0] - r1
                if np.any(dist > 0):
                    target_r = same_col[dist > 0, 0].min()
                    if np.all(inp[r1+1:target_r, c1] == bg):
                        out[r1+1:target_r, c1] = b1['color']
                if np.any(dist < 0):
                    target_r = same_col[dist < 0, 0].max()
                    if np.all(inp[target_r+1:r1, c1] == bg):
                        out[target_r+1:r1, c1] = b1['color']
                        
    return out

def solve_single_pixel_projection(solver) -> Optional[List[np.ndarray]]:
    """
    Finds single-pixel blobs and projects them towards other-colored blobs in the same row/col.
    """
    for inp, out in solver.pairs:
        if not np.array_equal(solve_one(inp), out):
            return None
            
    return [solve_one(ti) for ti in solver.test_in]
