from cellular import unupdate_cellular

import numpy as np

import config

MAX_UNUPDATES = 100
current_unupdates = 0

def isEnd(current_state):
	global current_unupdates
	current_unupdates += 1
	if current_unupdates >= MAX_UNUPDATES:
		return True
	return not (np.logical_and(current_state > 0 & current_state < outrange_standard_val)).any()

def ungrow_tumor(current_state, density_organ_state, save_frequency, kernel_size, steps, organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold, density_organ_map):
	i = 0
	while(not isEnd(current_state)):
		current_state = unupdate_cellular(current_state, density_organ_state, (kernel_size[0], kernel_size[1], kernel_size[2]), (organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold))
		temp = current_state.cpu().numpy().copy()
        # print(np.sum(temp==0))
		# all_states.append(temp)
		i += 1
		if i % save_frequency != 0:
			continue
		save(temp, run_id, i, steps, img, cropped_img, density_organ_map, save_path, min_x, max_x, min_y, max_y, min_z, max_z, threshold, organ_hu_lowerbound, organ_standard_val, outrange_standard_val, start_point)

def reverse(image, density_organ_state, ...):
	current_state = unmap(image)
	all_states = []
	tumor_out = ungrow_tumor(current_state, density_organ_state, kernel_size, steps, organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold, density_organ_map)
	return reverse_map_to_CT_values(tumor_out)

def temp_main():
	state = load_state(config.state_path)
	state[state == 0] = outrange_standard_val
	assert np.sum(state >= threshold) == 0
	save_frequency = 10
	ungrow_tumor(state, density_organ_state, save_frequency, kernel_size, steps, organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold, density_organ_map)

def main():
	temp_main()

if __name__ == "__main__":
	main()