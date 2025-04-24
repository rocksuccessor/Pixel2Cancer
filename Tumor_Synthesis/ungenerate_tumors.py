from cellular import unupdate_cellular
import SimpleITK as sitk

import numpy as np
import os
import time
import torch

import config
from constants import *

run_id = f"{config.file_id}{time.strftime('%Y%m%d%H%M%S')}"
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

    # Move tensor to CPU and convert to NumPy
    current_state_np = current_state.cpu().numpy()
    return not (np.logical_and(current_state_np > 0, current_state_np < outrange_standard_val)).any()

def map_to_CT_value(img, tumor_state, density_organ_map, steps, threshold, outrange_standard_val, organ_hu_lowerbound, organ_standard_val, start_point):

    img = img.astype(np.float32)
    tumor = tumor_state.astype(np.float32)
    density_organ_map = density_organ_map.astype(np.float32)



    # deal with the conflict vessel
    interval = (outrange_standard_val - organ_hu_lowerbound) / 3
    vessel_condition = (density_organ_map == outrange_standard_val) & (tumor >= threshold/2)
    vessel_value = np.random.randint(40, 50, tumor.shape, dtype=np.int16)

    # deal with the high intensity tissue
    high_tissue_condition = (density_organ_map == (organ_hu_lowerbound + 2 * interval)) & (tumor != 0)
    high_tissue_value = np.random.randint(20, 30, tumor.shape, dtype=np.int16)

    kernel = (3, 3)
    for z in range(vessel_value.shape[0]):
        vessel_value[z] = cv2.GaussianBlur(vessel_value[z], kernel, 0)
        high_tissue_value[z] = cv2.GaussianBlur(high_tissue_value[z], kernel, 0)

    img[vessel_condition] *= (organ_hu_lowerbound + interval/2) / outrange_standard_val
    img[high_tissue_condition] *= (organ_hu_lowerbound + 2 * interval) / outrange_standard_val

    

    # random tumor value
    tumor_value = np.random.randint(5, 15, tumor.shape, dtype=np.int16)
    tumor_value[tumor == 0] = 0

    # blur the tumor value
    kernel = (3, 3)
    for z in range(tumor_value.shape[0]):
        tumor_value[z] = cv2.GaussianBlur(tumor_value[z], kernel, 0)

    # CT mapping function
    # map_img = img * -(tumor/40 - 1) + tumor/50 * tumor_value

    bias = np.random.uniform(1,4)

    map_img = img * - (tumor / (threshold + bias * threshold) - 1) - (1 - tumor/(threshold + 0)) * tumor_value

    # postprocess
    tumor_region = map_img.copy()
    tumor_region[tumor == 0] = 0
    for z in range(tumor_region.shape[0]):
        tumor_region[z] = cv2.GaussianBlur(tumor_region[z], kernel, 0)
    map_img[(tumor >= threshold/2) & (density_organ_map >= (organ_hu_lowerbound + 2 * interval))] = tumor_region[(tumor >= threshold/2) & (density_organ_map >= (organ_hu_lowerbound + 2 * interval))]

    map_img = map_img.astype(np.int16)

    

    return map_img


def save(step_state, run_id, i, step, img, cropped_img, density_organ_map, save_path, min_x, max_x, min_y, max_y, min_z, max_z, threshold, organ_hu_lowerbound, organ_standard_val, outrange_standard_val, start_point, save_list):
    save_name = f"{run_id}{i}{step}.nii.gz"
    save = sitk.GetImageFromArray(step_state)
    sitk.WriteImage(save, os.path.join(save_path, 'tumor_reverse', save_name))

    img_out = map_to_CT_value(cropped_img, step_state, density_organ_map,
                            step, threshold, outrange_standard_val, organ_hu_lowerbound, organ_standard_val, start_point)


    save = sitk.GetImageFromArray(step_state)
    sitk.WriteImage(save, os.path.join(save_path, 'step_state_reverse', save_name))

    # # save the result
    img_save = img.copy()
    img_save[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1] = img_out
    save = sitk.GetImageFromArray(img_save)
    sitk.WriteImage(save, os.path.join(save_path, 'img_reverse', save_name))
    save_list.append(save_name)

    mask = np.zeros_like(img_save)
    mask[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1] = step_state
    mask[mask > 0] = 1
    save = sitk.GetImageFromArray(mask)
    sitk.WriteImage(save, os.path.join(save_path, 'mask_reverse', save_name))
    
    save_list.append(save_name)

def ungrow_tumor(current_state, density_organ_state, save_frequency, kernel_size, steps, organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold, density_organ_map, img, cropped_img, min_x, max_x, min_y, max_y, min_z, max_z, save_list):
	i = 0
	current_state = torch.tensor(sitk.GetArrayFromImage(current_state), dtype=torch.int32).cuda(device='cuda:0')
	assert current_state.shape == density_organ_map.shape
	while(not isEnd(current_state)):
		current_state = unupdate_cellular(current_state, density_organ_state, (kernel_size[0], kernel_size[1], kernel_size[2]), (organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold),0)
		current_state[current_state <= 0] = outrange_standard_val
		temp = current_state.cpu().numpy().copy()
        # print(np.sum(temp==0))
		# all_states.append(temp)
		i += 1
		if i % save_frequency != 0:
			continue
		save(temp, run_id, 0, i, img, cropped_img, density_organ_map, save_path, min_x, max_x, min_y, max_y, min_z, max_z, threshold, organ_hu_lowerbound, organ_standard_val, outrange_standard_val, None, save_list)

		print(f"step: {i}")
		unique_values, counts = np.unique(temp, return_counts=True)
		for value, count in zip(unique_values, counts):
			print(f"value: {value} times: {count} ")


# def reverse(image, density_organ_state, ...):
# 	current_state = unmap(image)
# 	all_states = []
# 	tumor_out = ungrow_tumor(current_state, density_organ_state, kernel_size, steps, organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold, density_organ_map)
# 	return reverse_map_to_CT_values(tumor_out)

def temp_main():
    state = load_state(config.state_path_for_reverse)
    # state[state > threshold] = threshold
    # assert np.sum(state > threshold) == 0 # temporarily disabled for speed of execution
    state[state == 0] = outrange_standard_val
    save_frequency = 10

    # img = sitk.ReadImage(file + '/ct.nii.gz')
    img = sitk.ReadImage(config.volume_path)
    img = sitk.GetArrayFromImage(img)
    mask = sitk.ReadImage(config.segmentation_path)
    mask = sitk.GetArrayFromImage(mask)

    # load organ and quantify
    # get the organ region
    organ_region = np.where(np.isin(mask, Organ_List['liver']))
    min_x = min(organ_region[0])
    max_x = max(organ_region[0])
    min_y = min(organ_region[1])
    max_y = max(organ_region[1])
    min_z = min(organ_region[2])
    max_z = max(organ_region[2])

    

    save_list = []
    ungrow_tumor(
        state, density_organ_state, save_frequency, kernel_size, steps,
        organ_hu_lowerbound, organ_standard_val, outrange_standard_val, threshold,
        density_organ_map, img, cropped_img, min_x, max_x, min_y, max_y, min_z, max_z, save_list
    )
    
def main():
        temp_main()

if __name__ == "__main__":
	main()
