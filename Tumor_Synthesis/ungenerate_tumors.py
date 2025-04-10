from cellular import unupdate_cellular

import numpy as np

MAX_UNUPDATES = 10
current_unupdates = 0

def isEnd(current_state):
	current_unupdates += 1
	if current_unupdates >= MAX_UNUPDATES:
		return True
	return not (np.logical_and(current_state > 0 & current_state < outrange_standard_val)).any()

def ungrow_tumor(current_state, density_organ_state, kernel_size, steps, all_states, organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold, density_organ_map):
	while(not isEnd(current_state)):
		current_state = unupdate_cellular(current_state, density_organ_state, (kernel_size[0], kernel_size[1], kernel_size[2]), (organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold))
		temp = current_state.cpu().numpy().copy()
        # print(np.sum(temp==0))
		all_states.append(temp)

def reverse(image, density_organ_state, ...):
	current_state = unmap(image)
	all_states = []
	tumor_out = ungrow_tumor(current_state, density_organ_state, kernel_size, steps, all_states, organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold, density_organ_map)
	return reverse_map_to_CT_values(tumor_out)
