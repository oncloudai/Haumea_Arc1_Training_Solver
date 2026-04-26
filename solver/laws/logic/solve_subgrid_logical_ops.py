
import numpy as np
from typing import List, Optional
from solver.utils import get_subgrids_by_bg

def solve_subgrid_logical_ops(solver) -> Optional[List[np.ndarray]]:
    # Try different background colors for dividing the grid
    for bg in range(10):
        subs_tr_list = [get_subgrids_by_bg(inp, bg) for inp, out in solver.pairs]
        if not all(len(subs) >= 2 for subs in subs_tr_list): continue
        
        sh, sw = subs_tr_list[0][0]['h'], subs_tr_list[0][0]['w']
        if not all(all(s['h'] == sh and s['w'] == sw for s in subs) for subs in subs_tr_list): continue

        # Try both (grid != sub_bg) and (grid == sub_bg)
        # What is sub_bg? Usually 0.
        for sub_bg in range(10):
            for condition in ['!=', '==']:
                # Masks for each training pair
                masks_tr = []
                for subs in subs_tr_list:
                    if condition == '!=':
                        masks_tr.append([(s['grid'] != sub_bg) for s in subs])
                    else:
                        masks_tr.append([(s['grid'] == sub_bg) for s in subs])
                
                # Try ops
                for op_name in ['AND', 'OR', 'XOR', 'NAND', 'NOR']:
                    # Check which color worked
                    for color in range(1, 10):
                        consistent = True
                        for idx, (inp_tr, out_tr) in enumerate(solver.pairs):
                            masks = masks_tr[idx]
                            if op_name == 'AND': res_mask = np.logical_and.reduce(masks)
                            elif op_name == 'OR': res_mask = np.logical_or.reduce(masks)
                            elif op_name == 'NAND': res_mask = ~np.logical_and.reduce(masks)
                            elif op_name == 'NOR': res_mask = ~np.logical_or.reduce(masks)
                            else: # XOR
                                res_mask = masks[0]
                                for m in masks[1:]: res_mask = np.logical_xor(res_mask, m)
                            
                            pred = np.zeros_like(out_tr)
                            pred[res_mask] = color
                            if not np.array_equal(pred, out_tr):
                                consistent = False; break
                        
                        if consistent:
                            # Apply to test
                            results = []
                            for ti in solver.test_in:
                                test_subs = get_subgrids_by_bg(ti, bg)
                                if len(test_subs) < 2: return None
                                if condition == '!=':
                                    test_masks = [(s['grid'] != sub_bg) for s in test_subs]
                                else:
                                    test_masks = [(s['grid'] == sub_bg) for s in test_subs]
                                    
                                if op_name == 'AND': res_mask = np.logical_and.reduce(test_masks)
                                elif op_name == 'OR': res_mask = np.logical_or.reduce(test_masks)
                                elif op_name == 'NAND': res_mask = ~np.logical_and.reduce(test_masks)
                                elif op_name == 'NOR': res_mask = ~np.logical_or.reduce(test_masks)
                                else:
                                    res_mask = test_masks[0]
                                    for m in test_masks[1:]: res_mask = np.logical_xor(res_mask, m)
                                
                                pred_test = np.zeros((sh, sw), dtype=int)
                                pred_test[res_mask] = color
                                results.append(pred_test)
                            return results
    return None
