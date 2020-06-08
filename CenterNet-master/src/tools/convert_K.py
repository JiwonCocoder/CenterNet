import numpy as np
from numpy.linalg import inv

#image_id = 0
loc_w = np.array([1.0473071503863416, 4.168695787212709, -0.24685933430989582])

loc_c = [loc_w[i] for i in [0, 2, 1]]
loc_c[1] *= -1
K = np.array([[529.5, 0, 365.0],[0, 529.5, 265.0],[0, 0, 1]])
Rtilt = np.array([[0.979589, 0.012593, -0.200614], [0.012593, 0.992231, 0.123772], [0.200614, -0.123772, 0.97182]])
tilted_loc_w = np.dot(Rtilt, loc_w)
inv_K = inv(K)
projected_loc_c =np.dot(K, np.transpose(loc_c))
unprojected_loc_c = np.dot(inv_K, projected_loc_c)

bbox2D = [328, 152, 346, 320]
bbox2D_ct= [(bbox2D[0] + bbox2D[2])/2,(bbox2D[1] + bbox2D[3])/2]
bbox2D_ct_loc = [bbox2D_ct[0]*loc_c[2], bbox2D_ct[1]*loc_c[2], loc_c[2]]
unprjected_bbox2D_ct_loc = np.dot(inv_K, bbox2D_ct_loc)

print(projected_loc_c)
print(unprojected_loc_c)
print(unprjected_bbox2D_ct_loc)