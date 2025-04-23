from cellular import unupdate_cellular
import SimpleITK as sitk

import numpy as np
import os
import time
import torch

import config
from constants import *

run_id = f"{config.file_id}_{time.strftime('%Y%m%d_%H%M%S')}"
MAX_UNUPDATES = 100
current_unupdates = 0
save_path = '/content/drive/MyDrive/dataset'
density_organ_map=sitk.ReadImage(config.save_path_density_organ_map)
density_organ_map = sitk.GetArrayFromImage(density_organ_map)
density_organ_state = torch.tensor(density_organ_map, dtype=torch.int32).cuda(device='cuda:0')
def load_state(path):
	img=sitk.ReadImage(path)
	return img

def isEnd(current_state):
	global current_unupdates
	current_unupdates += 1
	if current_unupdates >= MAX_UNUPDATES:
		return True
	return not (np.logical_and(current_state > 0, current_state < outrange_standard_val)).any()

def save(step_state, run_id, i, step):
	save_name = f"{run_id}_{i}_{step}.nii.gz"
	save = sitk.GetImageFromArray(step_state)
	sitk.WriteImage(save, os.path.join(save_path, 'step_state_reverse', save_name))

def ungrow_tumor(current_state, density_organ_state, save_frequency, kernel_size, steps, organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold, density_organ_map):
	i = 0
	while(not isEnd(current_state)):
		current_state = unupdate_cellular(current_state, density_organ_state, (kernel_size[0], kernel_size[1], kernel_size[2]), (organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold))
		current_state[current_state <= 0] = organ_standard_val
		temp = current_state.cpu().numpy().copy()
        # print(np.sum(temp==0))
		# all_states.append(temp)
		i += 1
		if i % save_frequency != 0:
			continue
		save(temp,run_id,0,i)


# def reverse(image, density_organ_state, ...):
# 	current_state = unmap(image)
# 	all_states = []
# 	tumor_out = ungrow_tumor(current_state, density_organ_state, kernel_size, steps, organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold, density_organ_map)
# 	return reverse_map_to_CT_values(tumor_out)

def temp_main():
	state = load_state(config.state_path_for_reverse)
	state[state > threshold] = threshold
	state[state == 0] = outrange_standard_val
	assert np.sum(state > threshold) == 0
	save_frequency = 10
	ungrow_tumor(state, density_organ_state, save_frequency, kernel_size, steps, organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold, density_organ_map)

def main():

	temp_main()

if __name__ == "__main__":
	main()
