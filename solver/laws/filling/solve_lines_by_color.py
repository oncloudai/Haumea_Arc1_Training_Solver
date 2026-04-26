import numpy as np

def solve_lines_by_color(solver):
    # Try two priorities: 'vh' (vertical then horizontal) and 'hv' (horizontal then vertical)
    for priority in ['vh', 'hv']:
        color_map = {} # color -> 'h' or 'v'
        consistent = True
        
        for inp, outp in solver.pairs:
            inp = np.array(inp)
            outp = np.array(outp)
            coords = np.argwhere(inp != 0)
            for r, c in coords:
                color = inp[r, c]
                h_count = np.sum(outp[r, :] == color)
                v_count = np.sum(outp[:, c] == color)
                
                if h_count > v_count:
                    if color in color_map and color_map[color] != 'h':
                        consistent = False; break
                    color_map[color] = 'h'
                elif v_count > h_count:
                    if color in color_map and color_map[color] != 'v':
                        consistent = False; break
                    color_map[color] = 'v'
                else:
                    # Ambiguous (h_count == v_count)
                    # For a single pixel, it could be both 1. 
                    # If it's already in map, fine. If not, we don't know yet.
                    pass
            if not consistent: break
        
        if not consistent or not color_map:
            continue

        # Verification step
        all_pairs_match = True
        for inp, outp in solver.pairs:
            pred = np.zeros_like(inp)
            coords = np.argwhere(inp != 0)
            
            if priority == 'vh':
                # Apply vertical first
                for r, c in coords:
                    color = inp[r, c]
                    if color_map.get(color) == 'v':
                        pred[:, c] = color
                # Apply horizontal
                for r, c in coords:
                    color = inp[r, c]
                    if color_map.get(color) == 'h':
                        pred[r, :] = color
            else:
                # Apply horizontal first
                for r, c in coords:
                    color = inp[r, c]
                    if color_map.get(color) == 'h':
                        pred[r, :] = color
                # Apply vertical
                for r, c in coords:
                    color = inp[r, c]
                    if color_map.get(color) == 'v':
                        pred[:, c] = color
            
            if not np.array_equal(pred, outp):
                all_pairs_match = False; break
        
        if all_pairs_match:
            # Apply to test inputs
            results = []
            for test_in in solver.test_in:
                test_in = np.array(test_in)
                test_output = np.zeros_like(test_in)
                coords = np.argwhere(test_in != 0)
                
                if priority == 'vh':
                    for r, c in coords:
                        color = test_in[r, c]
                        if color_map.get(color) == 'v':
                            test_output[:, c] = color
                    for r, c in coords:
                        color = test_in[r, c]
                        if color_map.get(color) == 'h':
                            test_output[r, :] = color
                else:
                    for r, c in coords:
                        color = test_in[r, c]
                        if color_map.get(color) == 'h':
                            test_output[r, :] = color
                    for r, c in coords:
                        color = test_in[r, c]
                        if color_map.get(color) == 'v':
                            test_output[:, c] = color
                results.append(test_output)
            return results
            
    return None
