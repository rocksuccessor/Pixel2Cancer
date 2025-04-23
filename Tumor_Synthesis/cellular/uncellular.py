import torch
from Cellular import _C

def unupdate_cellular(state_tensor, density_state_tensor, ranges, thresholds, flag, grow_per_cell=1, max_try=-1):
    if max_try < 0:
        max_try = grow_per_cell * 3
    return _CellularUnupdate.apply(ranges,grow_per_cell, max_try, state_tensor, density_state_tensor, thresholds, flag)

class _CellularUnupdate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                ranges,
                grow_per_cell,
                max_try,
                state_tensor,
                density_state_tensor,
                thresholds,
                flag
        ):
        Y_range, X_range, Z_range = ranges
        organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold = thresholds
        state_tensor_new = state_tensor.clone()
        _C.unupdate_cellular(state_tensor, density_state_tensor, Y_range, X_range, Z_range, grow_per_cell, max_try, organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold, flag, state_tensor_new)
        return state_tensor_new
    
