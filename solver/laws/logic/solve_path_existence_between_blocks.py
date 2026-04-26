
import numpy as np

def solve_path_existence_between_blocks(solver):
    def get_2x2_blocks(grid, color):
        h, w = grid.shape
        blocks = []
        for r in range(h - 1):
            for c in range(w - 1):
                if np.all(grid[r:r+2, c:c+2] == color):
                    blocks.append({(r, c), (r+1, c), (r, c+1), (r+1, c+1)})
        return blocks

    def has_path(grid, color, set1, set2):
        h, w = grid.shape
        mask = (grid == color)
        start_pixels = set()
        for r, c in set1:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]: start_pixels.add((nr, nc))
        target_pixels = set()
        for r, c in set2:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]: target_pixels.add((nr, nc))
        if not start_pixels or not target_pixels: return False
        visited = set(start_pixels)
        queue = list(start_pixels)
        while queue:
            r, c = queue.pop(0)
            if (r, c) in target_pixels: return True
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in visited:
                        visited.add((nr, nc)); queue.append((nr, nc))
        return False

    def apply_logic(inp):
        blocks = get_2x2_blocks(inp, 2)
        if len(blocks) != 2: return None
        path = has_path(inp, 8, blocks[0], blocks[1])
        return np.array([[8]]) if path else np.array([[0]])

    results = []
    for inp, outp in solver.pairs:
        pred = apply_logic(inp)
        if pred is None or not np.array_equal(pred, outp): return None
    for ti in solver.test_in:
        res = apply_logic(ti)
        if res is None: return None
        results.append(res)
    return results
