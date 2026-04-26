import numpy as np

def solve_most_frequent_3x3_object(solver):
    def get_3x3_shapes(grid):
        grid = np.array(grid)
        rows, cols = grid.shape
        shapes = {} # (color, shape_tuple) -> count
        
        for color in range(1, 10):
            visited = np.zeros_like(grid, dtype=bool)
            for r in range(rows):
                for c in range(cols):
                    if grid[r, c] == color and not visited[r, c]:
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
                                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == color and not visited[nr, nc]:
                                        visited[nr, nc] = True
                                        stack.append((nr, nc))
                        
                        min_r = min(r for r, c in coords)
                        max_r = max(r for r, c in coords)
                        min_c = min(c for r, c in coords)
                        max_c = max(c for r, c in coords)
                        
                        # Use a fixed 3x3 window starting from top-left
                        shape = np.zeros((3, 3), dtype=int)
                        for cr, cc in coords:
                            if cr - min_r < 3 and cc - min_c < 3:
                                shape[cr - min_r, cc - min_c] = color
                        
                        shape_tuple = tuple(shape.flatten())
                        key = (color, shape_tuple)
                        shapes[key] = shapes.get(key, 0) + 1
        return shapes

    results = []
    # No training verification needed if we follow the law? 
    # But it's good to check.
    
    for inp, out in solver.pairs:
        shapes = get_3x3_shapes(inp)
        if not shapes:
            return None
        sorted_shapes = sorted(shapes.items(), key=lambda x: (x[1], x[0]), reverse=True)
        # Check if the top one matches out
        best_shape = np.array(sorted_shapes[0][0][1]).reshape((3, 3))
        if not np.array_equal(best_shape, out):
            return None
            
    for ti in solver.test_in:
        shapes = get_3x3_shapes(ti)
        if not shapes:
            results.append(np.zeros((3, 3), dtype=int))
            continue
        sorted_shapes = sorted(shapes.items(), key=lambda x: (x[1], x[0]), reverse=True)
        best_shape = np.array(sorted_shapes[0][0][1]).reshape((3, 3))
        results.append(best_shape)
    return results
