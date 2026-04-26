
import numpy as np

def get_frame_components(grid):
    rows, cols = grid.shape
    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5 and (r, c) not in visited:
                comp = []
                q = [(r, c)]
                visited.add((r, c))
                while q:
                    curr_r, curr_c = q.pop(0)
                    comp.append((curr_r, curr_c))
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and \
                           grid[nr][nc] == 5 and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append((nr, nc))
                components.append({'color': 5, 'pixels': comp})
    return components

def get_hole_for_frame(grid, frame_pixels):
    min_r = min(p[0] for p in frame_pixels)
    max_r = max(p[0] for p in frame_pixels)
    min_c = min(p[1] for p in frame_pixels)
    max_c = max(p[1] for p in frame_pixels)
    
    hole_pixels = []
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid[r][c] == 0:
                hole_pixels.append((r, c))
    return hole_pixels

def get_shape(pixels):
    if not pixels: return set()
    min_r = min(p[0] for p in pixels)
    min_c = min(p[1] for p in pixels)
    return set((p[0] - min_r, p[1] - min_c) for p in pixels)

def get_objects_by_color(grid):
    objects = []
    other_colors = set(np.unique(grid)) - {0, 5}
    for color in other_colors:
        pixels = np.argwhere(grid == color)
        objects.append({
            'color': int(color),
            'pixels': [tuple(p) for p in pixels]
        })
    return objects

def apply_move_logic(input_grid):
    grid = np.array(input_grid)
    rows, cols = grid.shape
    
    frames = get_frame_components(grid)
    frame_holes = []
    for frame in frames:
        hole_pixels = get_hole_for_frame(grid, frame['pixels'])
        if hole_pixels:
            frame_holes.append({
                'frame': frame,
                'hole_pixels': hole_pixels,
                'shape': get_shape(hole_pixels)
            })
    
    objects = get_objects_by_color(grid)
    output_grid = grid.copy()
    used_objects = set()
    
    for fh in frame_holes:
        hole_shape = fh['shape']
        for i, obj in enumerate(objects):
            if i in used_objects: continue
            obj_shape = get_shape(obj['pixels'])
            if obj_shape == hole_shape:
                min_hole_r = min(p[0] for p in fh['hole_pixels'])
                min_hole_c = min(p[1] for p in fh['hole_pixels'])
                
                min_obj_r = min(p[0] for p in obj['pixels'])
                min_obj_c = min(p[1] for p in obj['pixels'])
                
                for r, c in obj['pixels']:
                    output_grid[r][c] = 0
                
                for r, c in obj['pixels']:
                    dr, dc = r - min_obj_r, c - min_obj_c
                    output_grid[min_hole_r + dr][min_hole_c + dc] = obj['color']
                
                used_objects.add(i)
                break
                
    return output_grid

def solve_move_objects_into_matching_holes(solver):
    results = []
    # Verification
    for inp, outp in solver.pairs:
        pred = apply_move_logic(inp)
        if not np.array_equal(pred, outp):
            return None
            
    for ti in solver.test_in:
        res = apply_move_logic(ti)
        results.append(res)
    return results
