import numpy as np

def solve_diagonal_propagation_from_marker(solver):
    def apply_logic(inp):
        inp = np.array(inp)
        h, w = inp.shape
        coords = np.argwhere(inp != 0)
        if len(coords) == 0: return inp
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        
        # Assume 2x2
        if r_max - r_min != 1 or c_max - c_min != 1: return None
        subgrid = inp[r_min:r_max+1, c_min:c_max+1]
        
        drawing_colors = [c for c in np.unique(subgrid) if c != 0 and c != 2]
        if not drawing_colors: return None
        color = drawing_colors[0]
        
        outp = np.zeros_like(inp)
        
        initial_shape = []
        for r in range(r_min, r_max+1):
            for c in range(c_min, c_max+1):
                if inp[r, c] != 0:
                    initial_shape.append((r, c))
                    if inp[r, c] != 2:
                        outp[r, c] = color
        
        directions = []
        if subgrid[0, 0] == 2: directions.append((-1, -1))
        if subgrid[0, 1] == 2: directions.append((-1, 1))
        if subgrid[1, 0] == 2: directions.append((1, -1))
        if subgrid[1, 1] == 2: directions.append((1, 1))
        
        if not directions: return None

        for dr, dc in directions:
            curr_shape = initial_shape
            while True:
                new_shape = []
                any_in_bounds = False
                for r, c in curr_shape:
                    nr, nc = r + dr, c + dc
                    new_shape.append((nr, nc))
                    if 0 <= nr < h and 0 <= nc < w:
                        any_in_bounds = True
                
                if any_in_bounds:
                    for nr, nc in new_shape:
                        if 0 <= nr < h and 0 <= nc < w:
                            outp[nr, nc] = color
                    curr_shape = new_shape
                else:
                    break
        return outp

    results = []
    # Verification
    for inp, outp in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, outp):
            return None
            
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
