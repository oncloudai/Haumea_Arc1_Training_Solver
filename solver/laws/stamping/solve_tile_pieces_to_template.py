import numpy as np
from typing import List, Optional

def solve_grid_a8c38be5(input_grid):
    # Template T (extracted from examples)
    T = np.array([
        [0,0,5,0,0,0,5,0,0],
        [0,5,5,5,0,5,5,5,0],
        [5,5,5,5,5,5,5,5,5],
        [0,5,5,5,5,5,5,5,0],
        [0,0,5,5,5,5,5,0,0],
        [0,5,5,5,5,5,5,5,0],
        [5,5,5,5,5,5,5,5,5],
        [0,5,5,5,0,5,5,5,0],
        [0,0,5,0,0,0,5,0,0]
    ])
    
    def get_connected_components(grid, colors_to_include):
        h, w = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        components = []
        for r in range(h):
            for c in range(w):
                if not visited[r, c] and grid[r, c] in colors_to_include:
                    color = grid[r, c]
                    component = []
                    stack = [(r, c)]
                    visited[r, c] = True
                    while stack:
                        curr_r, curr_c = stack.pop()
                        component.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == color:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                    min_r = min(p[0] for p in component)
                    min_c = min(p[1] for p in component)
                    rel_pixels = [(p[0] - min_r, p[1] - min_c) for p in component]
                    components.append({'color': color, 'pixels': rel_pixels})
        return components

    def solve_tiling(template, pieces):
        th, tw = template.shape
        # Total pixels check
        target_pixels = np.sum(template == 0)
        source_pixels = sum(len(p['pixels']) for p in pieces)
        if target_pixels != source_pixels: return None

        pieces.sort(key=lambda p: len(p['pixels']), reverse=True)
        grid = template.copy()
        
        state_count = 0
        max_states = 10000

        def backtrack(piece_idx):
            nonlocal state_count
            state_count += 1
            if state_count > max_states: return False

            if piece_idx == len(pieces): return True
            first_empty = None
            for r in range(th):
                for c in range(tw):
                    if grid[r, c] == 0:
                        first_empty = (r, c)
                        break
                if first_empty: break
            if not first_empty: return False
            er, ec = first_empty
            
            # Optimization: only try pieces that haven't been used
            for p_idx in range(piece_idx, len(pieces)):
                # Try piece p_idx at current position piece_idx
                pieces[piece_idx], pieces[p_idx] = pieces[p_idx], pieces[piece_idx]
                piece = pieces[piece_idx]
                
                # Piece MUST cover first_empty. Try all its pixels as the filler for er, ec.
                for pr, pc in piece['pixels']:
                    base_r, base_c = er - pr, ec - pc
                    can_place = True
                    for ppr, ppc in piece['pixels']:
                        nr, nc = base_r + ppr, base_c + ppc
                        if not (0 <= nr < th and 0 <= nc < tw and grid[nr, nc] == 0):
                            can_place = False; break
                    if can_place:
                        for ppr, ppc in piece['pixels']: grid[base_r + ppr, base_c + ppc] = piece['color']
                        if backtrack(piece_idx + 1): return True
                        for ppr, ppc in piece['pixels']: grid[base_r + ppr, base_c + ppc] = 0
                
                pieces[piece_idx], pieces[p_idx] = pieces[p_idx], pieces[piece_idx]
            return False

        if backtrack(0): return grid
        return None

    input_grid = np.array(input_grid)
    colors = np.unique(input_grid)
    non_bg_colors = [c for c in colors if c != 0 and c != 5]
    pieces = get_connected_components(input_grid, non_bg_colors)
    result = solve_tiling(T, pieces)
    return result if result is not None else T

def solve_tile_pieces_to_template(solver) -> Optional[List[np.ndarray]]:
    consistent = True
    for inp, out in solver.pairs:
        res = solve_grid_a8c38be5(inp)
        if not np.array_equal(res, out):
            consistent = False; break
            
    if consistent:
        return [solve_grid_a8c38be5(ti) for ti in solver.test_in]
    return None
