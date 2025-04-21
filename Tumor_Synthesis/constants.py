kernel_size = (3, 3, 3)
steps = 50  # step
Organ_HU = {'liver': [100, 160]}
organ_hu_lowerbound = Organ_HU['liver'][0] 
organ_standard_val = 0  # organ standard value
outrange_standard_val = Organ_HU['liver'][1]
threshold = 10  # threshold
