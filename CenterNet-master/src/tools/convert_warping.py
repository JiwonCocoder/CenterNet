import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
import _init_paths
from utils.ddd_utils import project_to_image, project_to_image_sun2, project_to_image_sun, alpha2rot_y, \
    compute_box_3d_sun, compute_box_3d_sun_2, compute_box_3d_sun_3, compute_box_3d_sun_4, compute_box_3d_sun_5, \
    compute_box_3d_sun_6,compute_box_3d_sun_8, compute_box_3d_sun_10
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d, draw_box_3d_sun
from utils.ddd_utils import rot_y2alpha, order_rotation_y_matrix, draw_box_3d_world
def read_Rtilt(Rtilt_path):
    f = open(Rtilt_path, 'r')
    for i, line in enumerate(f):
        # if i == 2:  # P2에 대해서 \n제거하고, 맨 앞 item(P2)이름빼고, 빼고 나머지 값들을 배열로
        # calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
        line = line[1:-1].split(',')
        Rtilt = np.array(line, dtype=np.float32)
        Rtilt = Rtilt.reshape(3, 3)  # 그냥 앞에서부터 순서대로 잘라서 넣는다.
        return Rtilt

def read_clib(calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        # if i == 2:  # P2에 대해서 \n제거하고, 맨 앞 item(P2)이름빼고, 빼고 나머지 값들을 배열로
        # calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
        line = line[1:-1].split(',')
        calib = np.array(line, dtype=np.float32)
        calib = calib.reshape(3, 4)  # 그냥 앞에서부터 순서대로 잘라서 넣는다.
        return calib


def compute_box_rtilt_xzy(dim, location, rotation_y, Rtilt):
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  rot_inds = order_rotation_y_matrix(R)
  rot_mat = R[rot_inds, :]
  x = dim[0]
  y = dim[1]
  z = dim[2]
  x_corners = [-x / 2, x / 2, x / 2, -x / 2, -x / 2, x / 2, x / 2, -x / 2]
  y_corners = [y / 2, y / 2, y / 2, y / 2, -y / 2, -y / 2, -y / 2, -y / 2]
  z_corners = [z / 2, z / 2, -z / 2, -z / 2, z / 2, z / 2, -z / 2, -z / 2]
  corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
  print("corners")
  print(corners)
  corners_3d = np.dot(rot_mat, corners)  # R_rot *x_ref_coord(camera_coordinate에서의 좌표값들)
  temp = np.array(location, dtype=np.float32).reshape(3, 1)
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3,1)  #(3,8)

  Rtilt = np.transpose(Rtilt)

  Rtilt_revised = np.zeros((3, 3))
  Rtilt_revised[0, 0] = Rtilt[0, 0]
  Rtilt_revised[0, 1] = -Rtilt[0, 2]
  Rtilt_revised[0, 2] = Rtilt[0, 1]

  Rtilt_revised[1, 0] = -Rtilt[2, 0]
  Rtilt_revised[1, 1] = Rtilt[2, 2]
  Rtilt_revised[1, 2] = -Rtilt[2, 1]

  Rtilt_revised[2, 0] = Rtilt[1, 0]
  Rtilt_revised[2, 1] = -Rtilt[1, 2]
  Rtilt_revised[2, 2] = Rtilt[1, 1]


  inversed_Rtilt_revised = inv(Rtilt_revised)
  # location_inversed_Rtilt_revised = np.dot(inversed_Rtilt_revised, location)
  # corners_3d = corners_3d + np.array(location_inversed_Rtilt_revised, dtype=np.float32).reshape(3, 1)
  corners_3d = np.dot(Rtilt_revised, corners_3d)
  # corners_3d = corners_3d.transpose(1, 0)
  # corners_3d = np.dot(Rtilt, corners_3d)
  return corners_3d.transpose(1, 0)

def compute_box_3d(dim_changed, location_changed, rotation_y, image_id):
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    c_minus, s_minus = np.cos(-rotation_y), np.sin(-rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    R_minus = np.array([[c_minus, 0, s_minus], [0, 1, 0], [-s_minus, 0, c_minus]], dtype=np.float32)
    # if (-abs(R[0, 0]) > -abs(R[2, 0])):
    #     rot_inds = [2, 1, 0]
    #     changed_RotIndex_list.append(int(image_id))
    # else:
    #     rot_inds = [1, 2, 0]
    #
    # rot_mat = R[:, rot_inds]

    rot_mat = R
    print(rot_mat)
    x = dim_changed[0]
    y = dim_changed[1]
    z = dim_changed[2]
    x_corners = [-x/2, x/2, x/2, -x/2, -x/2, x/2, x/2, -x/2]
    y_corners = [-y/2, -y/2, -y/2, -y/2, y/2, y/2, y/2, y/2]
    z_corners = [z/2, z/2, -z/2, -z/2, z/2, z/2, -z/2, -z/2]

    # x_corners = [x/2, x/2, -x/2, -x/2, x/2, x/2, -x/2, -x/2]
    # # y_corners = [y/2, y/2, y/2, y/2, -y/2, -y/2, -y/2, -y/2]
    # y_corners = [0, 0, 0, 0, -y, -y, -y, -y]
    # z_corners = [z/2, -z/2, -z/2, z/2, z/2, -z/2, -z/2, z/2]
    #
    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(rot_mat, corners)  # R_rot *x_ref_coord(camera_coordinate에서의 좌표값들)
    temp = np.array(location_changed, dtype=np.float32).reshape(3, 1)
    corners_3d = corners_3d + np.array(location_changed, dtype=np.float32).reshape(3, 1)  # camera좌표 (0,0)에서 시작했었으니까, 물체 center point로 평행이동시킴
    return corners_3d.transpose(1, 0)

img_path = 'C:\\obj_detection\\CenterNet-master\\CenterNet-master\\data\\SUNRGBD\\imgIndexing_all\\'
Rtilt_path = 'C:\\obj_detection\\CenterNet-master\\CenterNet-master\\data\\SUNRGBD\\RtiltIndexing\\'
id_path = 'C:\\obj_detection\\CenterNet-master\\CenterNet-master\\data\\SUNRGBD\\allDataIndexNew.txt'
ann_path = 'C:\\obj_detection\\CenterNet-master\\CenterNet-master\\data\\SUNRGBD\\labelIndexingNew\\'
calib_path = 'C:\\obj_detection\\CenterNet-master\\CenterNet-master\\data\\SUNRGBD\\calibIndexing\\'
f = open(id_path, 'r')
contents = f.read()
file_list = contents.split('\n')
print(file_list)
for i in range(len(file_list)):
    image = cv2.imread(img_path + file_list[i] +'.jpg')
    Rtilt = read_Rtilt(Rtilt_path + file_list[i] + '.txt')
#     calib = read_clib(calib_path + '{}.txt'.format(file_list[i]))
#     f_label = open(ann_path + '{}.txt'.format(file_list[i]))
#     for ann_ind, txt in enumerate(f_label):
#         tmp = txt[1:-2].split(',')
#         cat_id = tmp[0][1:-1]
#         location = [float(tmp[1]), float(tmp[2]), float(tmp[3])]
#         dim = [float(tmp[4]), float(tmp[5]), float(tmp[6])]  # 3D object dimensions: x길이, y길이, z길이
#         rotation_y = float(tmp[7])  # Rotation around Y-axis in camera coords. [-Pi; Pi]
#         bbox2D = [float(tmp[8]), float(tmp[9]), float(tmp[10]), float(tmp[11])]
#         alpha = float(tmp[12])
#
#         sun_permutation = [0, 2, 1]
#         dim_changed = [dim[i] * 2 for i in sun_permutation]  # 왜냐하면 dimension은 크기니까
#
#         # cam xzy를 이용해 Rtilt가 적용되지 않은 3d bbox를 그릴것
#         location_ori_changed = [location[i] for i in sun_permutation]
#         location_ori_changed[1] *= -1
#         # cam_xzy를 이용해 Rtilt가 적용된 3d bbox를 그릴것
#         location_tilt = np.dot(np.transpose(Rtilt), location)
#         location_changed = [location_tilt[i] for i in sun_permutation]
#         location_changed[1] *= -1
#         box_3d_rtilt = compute_box_rtilt_xzy(dim_changed, location_ori_changed, rotation_y, Rtilt)
#         box_2d_rtilt = project_to_image(box_3d_rtilt, calib)
#         image = draw_box_3d_world(image, box_2d_rtilt, file_list[i])

    #동일한 한 면을 잡아 그곳의 네 꼭지점에 대해서 투시변환 매트릭스를 구할것
    #
    cv2.imshow(file_list[i], image)
    # print(Rtilt)
    dst = cv2.warpPerspective(image, Rtilt, (448, 448))
    cv2.imshow("dd", dst)
    cv2.waitKey()
#     dst = cv2.warpPerspective(img, )