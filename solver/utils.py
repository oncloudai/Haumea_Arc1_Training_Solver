import numpy as np
from scipy.ndimage import label, binary_fill_holes
from typing import List, Dict, Any, Tuple, Optional

def get_blobs(grid, background=0, connectivity=8):
    mask = (grid != background)
    structure = np.ones((3, 3)) if connectivity == 8 else [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    labeled, n = label(mask, structure=structure)
    blobs = []
    for i in range(1, n + 1):
        coords = np.argwhere(labeled == i)
        min_rc = coords.min(axis=0)
        shape_mask = coords - min_rc
        blobs.append({
            'color': int(grid[coords[0][0], coords[0][1]]),
            'coords': coords,
            'top_left': min_rc,
            'normalized': shape_mask.tolist(),
            'size': len(coords)
        })
    return blobs

def get_homogeneous_blobs(grid, background=0, connectivity=8):
    blobs = []
    for color in range(1, 10):
        if color == background: continue
        mask = (grid == color)
        structure = np.ones((3,3)) if connectivity == 8 else [[0,1,0],[1,1,1],[0,1,0]]
        labeled, n = label(mask, structure=structure)
        for i in range(1, n+1):
            coords = np.argwhere(labeled == i)
            blobs.append({
                'color': color, 
                'size': len(coords), 
                'coords': coords, 
                'top_left': coords.min(axis=0)
            })
    return blobs

def get_holes(grid, background=0):
    h, w = grid.shape
    mask = (grid == background)
    labeled, n = label(mask, structure=[[0,1,0],[1,1,1],[0,1,0]])
    holes = []
    for i in range(1, n+1):
        coords = np.argwhere(labeled == i)
        on_boundary = False
        for r, c in coords:
            if r == 0 or r == h-1 or c == 0 or c == w-1:
                on_boundary = True; break
        if not on_boundary:
            holes.append({'size': len(coords), 'coords': coords})
    return holes

def get_enclosed_holes(grid, color):
    mask = (grid == color)
    filled = binary_fill_holes(mask)
    holes_mask = filled ^ mask
    labeled, n = label(holes_mask)
    holes = []
    for i in range(1, n+1):
        coords = np.argwhere(labeled == i)
        holes.append({'size': len(coords), 'coords': coords})
    return holes

def get_subgrids_by_bg(grid, bg=None):
    h, w = grid.shape
    if bg is None:
        # If bg is None, find any row/col that is uniform
        empty_rows = np.array([len(np.unique(grid[r])) == 1 for r in range(h)])
        empty_cols = np.array([len(np.unique(grid[:, c])) == 1 for c in range(w)])
    else:
        # If bg is provided, rows/cols must be ONLY that bg
        empty_rows = np.array([np.all(grid[r] == bg) for r in range(h)])
        empty_cols = np.array([np.all(grid[:, c] == bg) for c in range(w)])
    
    def get_regions(mask):
        regions = []; start = None
        for i, val in enumerate(mask):
            if not val: # Not a delimiter
                if start is None: start = i
            else: # Delimiter
                if start is not None:
                    regions.append((start, i))
                    start = None
        if start is not None: regions.append((start, len(mask)))
        return regions

    row_regions = get_regions(empty_rows)
    col_regions = get_regions(empty_cols)
    
    subgrids = []
    for r1, r2 in row_regions:
        for c1, c2 in col_regions:
            sub = grid[r1:r2, c1:c2]
            subgrids.append({
                'r': r1, 'c': c1, 'h': r2-r1, 'w': c2-c1,
                'grid': sub, 'colors': set(np.unique(sub).tolist())
            })
    return subgrids
