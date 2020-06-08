import numpy as np
import itertools
import math

bathtub_Aug_list = [['12391', -0.5815, [[0.15125539225931806, 0.6478929810244685, 0.029066218519257003], [0.334116810329967, 0.11695306177915454, 0.00445145209999992], 0.07858857636697869]], ['19582', 0.9623, [[0.14316203186857968, 0.4081545604834962, 0.0015533349826490073], [0.09413490687470327, 0.4297739501192932, 0.1326985467045454], 0.22513581707233365]], ['12510', 0.6183, [[0.09897026354656657, 0.415316675151157, 0.012717341543591665], [0.33334170929269025, 0.2706553432961104, 0.1711594312000001], 0.4863709429847274]], ['17706', -0.5563, [[0.03905801501403472, 0.41426684435715044, 0.09930555271613112], [0.45712348035012274, 0.07827504125537377, 0.02507060459999999], 0.12492088884011254]], ['12491', 0.5383, [[0.11175396428018852, 0.496670040947218, 0.04785351855199968], [0.052579400310069135, 0.46484976767965436, 0.13597031009999988], 0.2234358416469565]]]

bathtub_not_Aug_list = [['12391', -0.5815, [[0.20862433225931803, 0.20421218102446836, 0.02489307148074299], [0.3436902103299668, 0.0040771482208454835, 0.07964690790000006], 0.3493496763669788]]]


base_error_list = []
aug_error_list = []
for base in bathtub_not_Aug_list:
    for aug in bathtub_Aug_list:
        #만약 aug에 동일한게 있다면, (image_id 랑 loc_gt_w_x가 동일한 상황)
        if base[0] == aug[0] and base[1]  == aug[1]:
            base_error_list.append([base[2][0][0], base[2][0][1], base[2][0][2],
                                    base[2][1][0], base[2][1][1], base[2][1][2],
                                    base[2][2]])
            aug_error_list.append([aug[2][0][0], aug[2][0][1], aug[2][0][2],
                                    aug[2][1][0], aug[2][1][1], aug[2][1][2],
                                    aug[2][2]])

print(len(base_error_list))
base_error_array = np.array(base_error_list)
aug_error_array = np.array(aug_error_list)
base_error_array[:, 6] = np.rint((base_error_array[:, 6] * 180 / math.pi))
aug_error_array[:, 6] = np.rint((aug_error_array[:, 6] * 180 / math.pi))

base_error_mean_array = np.round(np.mean(base_error_array, axis = 0),4)
aug_error_mean_array = np.round(np.mean(aug_error_array, axis = 0),4)

print(base_error_mean_array)
print(aug_error_mean_array)