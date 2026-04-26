
import numpy as np

def solve_subgrid_inside_frame(solver):
    # This law extracts the content inside the first hollow rectangular frame it finds.
    results = []
    
    for ti in solver.test_in:
        ti = np.array(ti)
        found_content = None
        colors = np.unique(ti)
        # Try colors in descending order of frequency or just any color that forms a frame
        for color in colors:
            if color == 0: continue
            coords = np.argwhere(ti == color)
            if len(coords) < 4: continue
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            h, w = r_max - r_min + 1, c_max - c_min + 1
            
            # Check if it's a hollow rectangle frame
            if len(coords) == 2*h + 2*w - 4:
                if h > 2 and w > 2:
                    found_content = ti[r_min+1:r_max, c_min+1:c_max]
                    break
        
        if found_content is not None:
            results.append(found_content)
        else:
            return None
            
    # Verification against training
    for inp, outp in solver.pairs:
        inp = np.array(inp)
        outp = np.array(outp)
        match = False
        colors = np.unique(inp)
        for color in colors:
            if color == 0: continue
            coords = np.argwhere(inp == color)
            if len(coords) < 4: continue
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            h, w = r_max - r_min + 1, c_max - c_min + 1
            if len(coords) == 2*h + 2*w - 4:
                if h > 2 and w > 2:
                    inside = inp[r_min+1:r_max, c_min+1:c_max]
                    if inside.shape == outp.shape and np.array_equal(inside, outp):
                        match = True
                        break
        if not match:
            return None

    return results
