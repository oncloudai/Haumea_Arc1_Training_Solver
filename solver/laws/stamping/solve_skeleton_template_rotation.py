import numpy as np

def solve_skeleton_template_rotation(solver):
    def get_skeletons(grid, skeleton_color=4):
        grid = np.array(grid)
        visited = np.zeros_like(grid, dtype=bool)
        skeletons = []
        rows, cols = grid.shape
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == skeleton_color and not visited[r, c]:
                    coords = []
                    stack = [(r, c)]
                    visited[r, c] = True
                    while stack:
                        curr_r, curr_c = stack.pop()
                        coords.append((curr_r, curr_c))
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0: continue
                                nr, nc = curr_r + dr, curr_c + dc
                                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == skeleton_color and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    stack.append((nr, nc))
                    skeletons.append(sorted(coords))
        return skeletons

    def get_symmetry(points, i):
        transformed = []
        for r, c in points:
            if i == 0: nr, nc = r, c
            elif i == 1: nr, nc = c, -r
            elif i == 2: nr, nc = -r, -c
            elif i == 3: nr, nc = -c, r
            elif i == 4: nr, nc = r, -c
            elif i == 5: nr, nc = -r, c
            elif i == 6: nr, nc = c, r
            elif i == 7: nr, nc = -c, -r
            transformed.append((nr, nc))
        return transformed

    def normalize(points):
        if not points: return [], 0, 0
        min_r = min(r for r, c in points)
        min_c = min(c for r, c in points)
        return sorted([(r - min_r, c - min_c) for r, c in points]), min_r, min_c

    def run_one(input_grid, skel_color):
        input_grid = np.array(input_grid)
        skeletons = get_skeletons(input_grid, skel_color)
        if len(skeletons) < 2:
            return None

        skel_neighbors = [{} for _ in skeletons]
        rows, cols = input_grid.shape
        for r in range(rows):
            for c in range(cols):
                color = input_grid[r, c]
                if color != 0 and color != skel_color:
                    min_dist = float('inf')
                    best_skel_idx = -1
                    for i, skel in enumerate(skeletons):
                        for sr, sc in skel:
                            dist = max(abs(r - sr), abs(c - sc))
                            if dist < min_dist:
                                min_dist = dist
                                best_skel_idx = i
                    if best_skel_idx != -1 and min_dist <= 3:
                        skel_min_r = min(sr for sr, sc in skeletons[best_skel_idx])
                        skel_min_c = min(sc for sr, sc in skeletons[best_skel_idx])
                        skel_neighbors[best_skel_idx][(r - skel_min_r, c - skel_min_c)] = color

        skel_data = []
        for i, skel in enumerate(skeletons):
            norm_skel, min_r, min_c = normalize(skel)
            skel_data.append({
                'norm_coords': norm_skel,
                'min_r': min_r,
                'min_c': min_c,
                'neighbors': skel_neighbors[i]
            })

        ref_idx = -1
        max_neighbors = -1
        for i, data in enumerate(skel_data):
            if len(data['neighbors']) > max_neighbors:
                max_neighbors = len(data['neighbors'])
                ref_idx = i
        
        if ref_idx == -1 or max_neighbors == 0:
            return None
            
        ref_data = skel_data[ref_idx]
        output_grid = input_grid.copy()
        any_filled = False

        for i, data in enumerate(skel_data):
            if i == ref_idx:
                continue
            
            valid_transforms = []
            for t in range(8):
                transformed_skel = get_symmetry(ref_data['norm_coords'], t)
                norm_transformed_skel, t_min_r, t_min_c = normalize(transformed_skel)
                if norm_transformed_skel == data['norm_coords']:
                    transformed_neighbors = {}
                    for (rel_r, rel_c), color in ref_data['neighbors'].items():
                        tr, tc = get_symmetry([(rel_r, rel_c)], t)[0]
                        transformed_neighbors[(tr - t_min_r, tc - t_min_c)] = color
                    
                    match = True
                    for rel_pos, color in data['neighbors'].items():
                        if transformed_neighbors.get(rel_pos, 0) != color:
                            match = False
                            break
                    if match:
                        valid_transforms.append((t, transformed_neighbors))

            if valid_transforms:
                valid_transforms.sort(key=lambda x: sum(1 for p, c in data['neighbors'].items() if x[1].get(p) == c), reverse=True)
                _, best_neighbors = valid_transforms[0]
                for (rel_r, rel_c), color in best_neighbors.items():
                    target_r = data['min_r'] + rel_r
                    target_c = data['min_c'] + rel_c
                    if 0 <= target_r < output_grid.shape[0] and 0 <= target_c < output_grid.shape[1]:
                        if output_grid[target_r, target_c] == 0:
                            output_grid[target_r, target_c] = color
                            any_filled = True
        
        return output_grid

    possible_colors = [c for c in range(1, 10)]
    for skel_color in possible_colors:
        # Try this color as skeleton
        results = []
        consistent = True
        for inp, out in solver.pairs:
            pred = run_one(inp, skel_color)
            if pred is None or not np.array_equal(pred, out):
                consistent = False
                break
        
        if consistent:
            test_results = []
            for ti in solver.test_in:
                res = run_one(ti, skel_color)
                if res is None:
                    consistent = False
                    break
                test_results.append(res)
            if consistent:
                return test_results
    return None
