import numpy as np
import itertools
import math

nightstand_Aug_list = [['12652', -1.5102, [[0.07014419255890081, 0.24949988363017628, 0.041912148137023236], [0.579636311302204, 0.06526283034162778, 0.1876893276999999], 0.09201765246493454]], ['12683', -1.4493, [[0.12545401993558025, 0.3927612302930674, 0.032866738568786635], [0.17216041858253867, 0.07636404952527798, 0.16373372789999996], 0.04390131090768956]], ['17908', 1.0269, [[0.08236382734135506, 0.4664237181805495, 0.08458056798112157], [0.24933374555906562, 0.08961346229493383, 0.0468739306], 0.05240107789917148]], ['12305', -0.2503, [[0.07745466617516858, 0.2662892822422118, 0.0497865283119531], [0.11789130869480635, 0.025492500121027384, 0.08769482389999994], 0.04305131164250953]], ['12305', 1.5777, [[0.12705799449513577, 0.29421153878981077, 0.05980097491640189], [0.19758147900657724, 0.21592010753947116, 0.2636654261], 0.5149673052039789]], ['17943', 1.1606, [[0.03552538638704372, 0.14719714117671945, 0.038844868775219976], [0.32714312821138447, 0.16656819335856876, 0.06596970339999986], 0.2591978902192418]]]

nightstand_not_Aug_list = [['12683', -1.4493, [[0.10708351993558018, 0.41225913029306716, 0.017388881431213365], [0.006265738582538649, 0.08340698952527803, 0.04865981210000003], 0.3518661890923105]], ['12305', -0.2503, [[0.043139756175168575, 0.5673384177577878, 0.02767096168804689], [0.14256856869480639, 0.04159723987897268, 0.30814844389999996], 0.32401481164250945]]]

base_error_list = []
aug_error_list = []
for base in nightstand_not_Aug_list:
    for aug in nightstand_Aug_list:
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

# base_error_mean_array = np.round(np.mean(base_error_array, axis = 0),4)
# aug_error_mean_array = np.round(np.mean(aug_error_array, axis = 0),4)
#
# print(base_error_mean_array)
# print(aug_error_mean_array)

base_error_mean_array = np.round(np.sum(base_error_array, axis = 0),4)
aug_error_mean_array = np.round(np.sum(aug_error_array, axis = 0),4)

print(base_error_mean_array.tolist())
print(aug_error_mean_array.tolist())