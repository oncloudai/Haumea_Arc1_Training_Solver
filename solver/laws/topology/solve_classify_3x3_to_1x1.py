
import numpy as np
from typing import List, Optional

def solve_classify_3x3_to_1x1(solver) -> Optional[List[np.ndarray]]:
    """
    Classify 3x3 input grids into 1x1 outputs based on some property.
    Property: Horizontal or Vertical Symmetry.
    """
    if not all(inp.shape == (3, 3) for inp, out in solver.pairs): return None
    if not all(out.shape == (1, 1) for inp, out in solver.pairs): return None
    
    # Possible properties
    def is_h_sym(g): return np.array_equal(g, g[:, ::-1])
    def is_v_sym(g): return np.array_equal(g, g[::-1, :])
    def is_any_sym(g): return is_h_sym(g) or is_v_sym(g)
    
    properties = [is_h_sym, is_v_sym, is_any_sym]
    
    for prop_func in properties:
        # Check if output is consistent for True and False cases of this property
        map_results = {} # property_value -> output_color
        consistent = True
        for inp, out in solver.pairs:
            p_val = prop_func(inp)
            out_color = out[0, 0]
            if p_val in map_results:
                if map_results[p_val] != out_color:
                    consistent = False; break
            else:
                map_results[p_val] = out_color
        
        if consistent and len(map_results) == 2:
            # Valid classifier found!
            results = []
            for ti in solver.test_in:
                p_val = prop_func(ti)
                if p_val in map_results:
                    results.append(np.array([[map_results[p_val]]]))
                else:
                    # If test input has a property value not seen in training, 
                    # we can't reliably predict.
                    consistent = False; break
            if consistent:
                return results
                
    return None
