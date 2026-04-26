import numpy as np
from typing import List, Optional
from scipy.ndimage import label

def solve_intra_input_template_completion(solver) -> Optional[List[np.ndarray]]:
    def get_objects(grid):
        grid = np.array(grid)
        mask = (grid != 0)
        labeled, num = label(mask)
        objects = []
        for i in range(1, num + 1):
            coords = np.argwhere(labeled == i)
            r_min, c_min = coords.min(axis=0)
            color = grid[coords[0][0], coords[0][1]]
            # Normalized shape
            shape = tuple(sorted([(r - r_min, c - c_min) for r, c in coords]))
            objects.append({
                'r_min': r_min, 'c_min': c_min,
                'color': color, 'shape': shape,
                'coords': coords
            })
        return objects

    def solve_single(input_grid):
        input_grid = np.array(input_grid)
        H, W = input_grid.shape
        output = input_grid.copy()
        
        objects = get_objects(input_grid)
        if not objects: return output
        
        # A "rule" is ((color1, shape1), (color2, shape2), dr, dc)
        # meaning if (c1, s1) is at (r, c), then (c2, s2) should be at (r+dr, c+dc)
        rules = []
        
        # 1. Discover rules from existing pairs
        for i in range(len(objects)):
            for j in range(len(objects)):
                if i == j: continue
                o1, o2 = objects[i], objects[j]
                dr, dc = o2['r_min'] - o1['r_min'], o2['c_min'] - o1['c_min']
                
                # Check if this offset is "local" (e.g. within 5x5)
                # Actually, in some tasks it might be further, but let's stick to local for now.
                if abs(dr) > 5 or abs(dc) > 5: continue
                
                rule = (o1['color'], o1['shape'], o2['color'], o2['shape'], dr, dc)
                if rule not in rules: rules.append(rule)
        
        # 2. Filter rules: they must be consistent and have "missing" instances
        valid_rules = []
        for r in rules:
            c1, s1, c2, s2, dr, dc = r
            
            # Find all instances of (c1, s1)
            instances1 = [o for o in objects if o['color'] == c1 and o['shape'] == s1]
            # Find all instances of (c2, s2)
            instances2 = [o for o in objects if o['color'] == c2 and o['shape'] == s2]
            
            # For every instance of (c1, s1), is (c2, s2) at (r+dr, c+dc)?
            has_gap = False
            possible = True
            for o in instances1:
                target_r, target_c = o['r_min'] + dr, o['c_min'] + dc
                # Is there an object (c2, s2) at (target_r, target_c)?
                exists = any(o2['r_min'] == target_r and o2['c_min'] == target_c 
                             and o2['color'] == c2 and o2['shape'] == s2 for o2 in instances2)
                if not exists:
                    has_gap = True
                    # Check for contradiction: is the target area empty?
                    for sr, sc in s2:
                        nr, nc = target_r + sr, target_c + sc
                        if 0 <= nr < H and 0 <= nc < W:
                            if output[nr, nc] != 0:
                                possible = False; break
                        else:
                            possible = False; break
                if not possible: break
            
            if possible and has_gap:
                valid_rules.append(r)
                
            # Symmetric check: for every instance of (c2, s2), is (c1, s1) at (r-dr, c-dc)?
            # (We'll handle this by the fact that we try all i, j pairs above)

        # 3. Apply valid rules
        changed = True
        while changed:
            changed = False
            for c1, s1, c2, s2, dr, dc in valid_rules:
                # Find current objects of type (c1, s1)
                current_objects = get_objects(output)
                instances1 = [o for o in current_objects if o['color'] == c1 and o['shape'] == s1]
                for o in instances1:
                    target_r, target_c = o['r_min'] + dr, o['c_min'] + dc
                    # Check if (c2, s2) already exists there
                    exists = any(o2['r_min'] == target_r and o2['c_min'] == target_c 
                                 and o2['color'] == c2 and o2['shape'] == s2 for o2 in current_objects)
                    if not exists:
                        # Fill it
                        for sr, sc in s2:
                            nr, nc = target_r + sr, target_c + sc
                            if 0 <= nr < H and 0 <= nc < W:
                                if output[nr, nc] == 0:
                                    output[nr, nc] = c2
                                    changed = True
        return output

    test_preds = []
    for inp, out in solver.pairs:
        pred = solve_single(inp)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return None
    
    for ti in solver.test_in:
        pred = solve_single(ti)
        if pred is None: return None
        test_preds.append(pred)
        
    return test_preds
