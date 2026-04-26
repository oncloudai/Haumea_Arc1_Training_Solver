import numpy as np
from typing import List, Optional

def solve_scaled_template_stamping_at_markers(solver) -> Optional[List[np.ndarray]]:
    """
    Logic for 6aa20dc0:
    1. Identify a 3x3 template that contains at least two infrequent colors.
    2. Identify markers of those two colors (squares of some size s).
    3. If two squares of the same size s match a relative position from any symmetry of the template, 
       stamp that 3x3 template at that location, scaled by s.
    """
    def get_symmetries(grid):
        soms = []
        curr = np.array(grid)
        for _ in range(4):
            soms.append(curr.copy())
            soms.append(np.fliplr(curr).copy())
            curr = np.rot90(curr)
        unique_soms = []
        seen = set()
        for s in soms:
            t = tuple(s.flatten())
            if t not in seen:
                unique_soms.append(s)
                seen.add(t)
        return unique_soms

    def find_squares(grid, color):
        rows, cols = grid.shape
        squares = []
        visited = np.zeros((rows, cols), dtype=bool)
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == color and not visited[r, c]:
                    s = 0
                    while r + s < rows and c + s < cols and grid[r+s, c] == color and grid[r, c+s] == color: s += 1
                    is_square = True
                    for dr in range(s):
                        for dc in range(s):
                            if r+dr >= rows or c+dc >= cols or grid[r+dr, c+dc] != color: is_square = False; break
                        if not is_square: break
                    if is_square:
                        squares.append((r, c, s))
                        for dr in range(s):
                            for dc in range(s): visited[r+dr, c+dc] = True
        return squares

    def apply(input_grid):
        input_grid = np.array(input_grid)
        rows, cols = input_grid.shape
        unique, counts = np.unique(input_grid, return_counts=True)
        bg_color = int(unique[np.argmax(counts)])
        best_template = None
        best_markers = None
        max_score = -1

        for r in range(rows - 2):
            for c in range(cols - 2):
                block = input_grid[r:r+3, c:c+3]
                block_unique = np.unique(block)
                block_infrequent = [int(bc) for bc in block_unique if bc != bg_color]
                if len(block_infrequent) >= 2:
                    for i in range(len(block_infrequent)):
                        for j in range(i + 1, len(block_infrequent)):
                            c1, c2 = block_infrequent[i], block_infrequent[j]
                            score = np.sum((block != bg_color) & (block != c1) & (block != c2))
                            if score > max_score:
                                max_score, best_template, best_markers = score, block.copy(), (c1, c2)

        if best_template is None: return input_grid
        marker1, marker2 = best_markers
        m1_poss, m2_poss = np.argwhere(best_template == marker1), np.argwhere(best_template == marker2)
        if len(m1_poss) == 0 or len(m2_poss) == 0: return input_grid

        s1_list, s2_list = find_squares(input_grid, marker1), find_squares(input_grid, marker2)
        output_grid, template_soms = input_grid.copy(), get_symmetries(best_template)
        
        for r1, c1, s1 in s1_list:
            for r2, c2, s2 in s2_list:
                if s1 == s2:
                    s = s1
                    for som in template_soms:
                        m1_ps, m2_ps = np.argwhere(som == marker1), np.argwhere(som == marker2)
                        if len(m1_ps) == 0 or len(m2_ps) == 0: continue
                        for m1p in m1_ps:
                            for m2p in m2_ps:
                                R0, C0 = r1 - m1p[0] * s, c1 - m1p[1] * s
                                if r2 == R0 + m2p[0] * s and c2 == C0 + m2p[1] * s:
                                    for br in range(3):
                                        for bc in range(3):
                                            for drr in range(s):
                                                for dcc in range(s):
                                                    rr, cc = R0+br*s+drr, C0+bc*s+dcc
                                                    if 0 <= rr < rows and 0 <= cc < cols: output_grid[rr, cc] = int(som[br, bc])
                                    break
        return output_grid

    for inp, out_expected in solver.pairs:
        pred = apply(inp)
        if not np.array_equal(pred, out_expected):
            return None
            
    results = []
    for ti in solver.test_in:
        results.append(apply(ti))
    return results
