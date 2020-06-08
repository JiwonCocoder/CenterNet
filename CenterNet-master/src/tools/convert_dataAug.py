from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image
from scipy import io
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib
import pickle
import math
import json
import numpy as np
import cv2

DATA_PATH = '../../data/SUNRGBD/'
# DEBUG = False
DEBUG = True
# VAL_PATH = DATA_PATH + 'training/label_val/'
import os

SPLITS = ['3dop', 'subcnn']
import _init_paths
from utils.ddd_utils import project_to_image, project_to_image_sun2, project_to_image_sun, alpha2rot_y, \
    compute_box_3d_sun, compute_box_3d_sun_2, compute_box_3d_sun_3, compute_box_3d_sun_4, compute_box_3d_sun_5, \
    compute_box_3d_sun_6,compute_box_3d_sun_8, compute_box_3d_sun_10
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d, draw_box_3d_sun
from utils.ddd_utils import rot_y2alpha, order_rotation_y_matrix, draw_box_3d_world
import pdb

'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
        (0)          'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
        (1)          truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
        (2)          0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
        (3)    
   4    bbox         2D bounding box of object in the image (0-based index):
        (4,5,6,7)     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
        (8,9,10)                 
   3    location     3D object location x,y,z in camera coordinates (in meters)
        (11,12,13)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
        (14)
   1    score        Only for results: Float, indicating confidence in
        (15)         detection, needed for p/r curves, higher is better.
'''


# def draw_box_3d(corners, c=(0, 0, 255)):
#     fig = pyplot.figure()
#     ax = Axes3D(fig)
#     face_idx = [[0,1,5,4], #앞
#               [1,2,6, 5], #왼
#               [2,3,7,6],  #뒤
#               [3,0,4,7]] #오
#     for ind_f in range(3, -1, -1):
#         f = face_idx[ind_f]
#             ax.plot([corners])
#           cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
#                    (corners[f[(j+1)%4], 0], corners[f[(j+1)%4], 1]), c, 2, lineType=cv2.LINE_AA)
#           # cv2.imshow("img", image)
#           # cv2.waitKey()
#
#         if ind_f == 0: #암면에 대해서는 대각선으로 표시
#           cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
#                    (corners[f[2], 0], corners[f[2], 1]), c, 1, lineType=cv2.LINE_AA)
#           cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
#                    (corners[f[3], 0], corners[f[3], 1]), c, 1, lineType=cv2.LINE_AA)
#     return image
#
#     ax.scatter(x3,z3,y3, c=bgr, depthshade = False)
#     pyplot.show()

def _bbox_to_coco_bbox(bbox):
    return [(bbox[0]), (bbox[1]),
            (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]


def read_clib(calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        # if i == 2:  # P2에 대해서 \n제거하고, 맨 앞 item(P2)이름빼고, 빼고 나머지 값들을 배열로
        # calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
        line = line[1:-1].split(',')
        calib = np.array(line, dtype=np.float32)
        calib = calib.reshape(3, 4)  # 그냥 앞에서부터 순서대로 잘라서 넣는다.
        return calib


def read_Rtilt(Rtilt_path):
    f = open(Rtilt_path, 'r')
    for i, line in enumerate(f):
        # if i == 2:  # P2에 대해서 \n제거하고, 맨 앞 item(P2)이름빼고, 빼고 나머지 값들을 배열로
        # calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
        line = line[1:-1].split(',')
        print(line)
        Rtilt = np.array(line, dtype=np.float32)
        Rtilt = Rtilt.reshape(3, 3)  # 그냥 앞에서부터 순서대로 잘라서 넣는다.
        return Rtilt


def read_imgIndex(imgIndex_path):
    f = open(imgIndex_path, 'r')
    index = []
    for line in enumerate(f):
        # if i == 2:  # P2에 대해서 \n제거하고, 맨 앞 item(P2)이름빼고, 빼고 나머지 값들을 배열로
        # calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
        line = line[0:-1]
        index.append(line)
    return index


'''
cats = ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck', 'Person_sitting',
        'Tram', 'Misc', 'DontCare']
'''

cats = ['counter', 'paper_cutter', 'papers', 'lamination_machine', 'box', 'drawer',
        'garbage_bin', 'scanner', 'poster', 'chair', 'tv', 'bed', 'night_stand', 'ottoman', 'dresser_mirror', 'dresser',
        'lamp', 'sofa', 'sofa_bed']
classes = [
    "bed",
    "books",
    "book",
    "bottle",
    "bowl",
    "plate",
    "box",
    "cabinet",
    "drawer",
    "kitchen_cabinet",
    "file_cabinet",
    "chair",
    "kitchen_counter",
    "counter",
    "cup",
    "mug",
    "desk",
    "door",
    "lamp",
    "night_stand",
    "painting",
    "picture",
    "pillow",
    "plant",
    "plants",
    "flowers",
    "shelf",
    "bookshelf",
    "sofa_chair",
    "sofa",
    "table",
    "endtable",
    "coffee_table",
    "dining_table",
    "toilet",
    "tv"
]
class_final = [
    "bed",
    "book",
    "diskware",
    "box",
    "drawer",
    "chair",
    "counter",
    "desk",
    "door",
    "lamp",
    "painting",
    "pillow",
    "plant",
    "shelf",
    "sofa",
    "table",
    "toilet",
    "tv"

]

skip_img = [50,
            51,
            55,
            115,
            116,
            123,
            125,
            155,
            246,
            285,
            306,
            388,
            409,
            679,
            715,
            716,
            739,
            778,
            802,
            818,
            852,
            918,
            976,
            1115,
            1131,
            1354,
            1365,
            1677,
            1679,
            1693,
            1694,
            1695,
            1696,
            1699,
            1705,
            1861,
            1879,
            1882,
            1884,
            1898,
            1900,
            1915,
            1948,
            2042,
            2060,
            2072,
            2087,
            2092,
            2093,
            2112,
            2113,
            2130,
            2141,
            2176,
            2180,
            2182,
            2186,
            2188,
            2194,
            2212,
            2214,
            2230,
            2232,
            2237,
            2238,
            2240,
            2303,
            2304,
            2342,
            2343,
            2393,
            2418,
            2428,
            2449,
            2498,
            2509,
            2619,
            2802,
            2852,
            2871,
            2919,
            2933,
            2977,
            3133,
            3223,
            3265,
            3283,
            3284,
            3300,
            3301,
            3323,
            3335,
            3432,
            3441,
            3456,
            3487,
            3488,
            3490,
            3493,
            3497,
            3514,
            3543,
            3545,
            3614,
            3615,
            3625,
            3700,
            3796,
            3808,
            3902,
            3970,
            4004,
            4017,
            4019,
            4026,
            4037,
            4047,
            4052,
            4053,
            4112,
            4113,
            4138,
            4167,
            4195,
            4219,
            4254,
            4266,
            4289,
            4312,
            4376,
            4378,
            4439,
            4522,
            4564,
            4628,
            4639,
            4642,
            4699,
            4847,
            4924,
            5022,
            5090,
            5091,
            5096,
            5102,
            5120,
            5204,
            5216,
            5530,
            5357,
            5377,
            5449,
            5756,
            5808,
            5809,
            5864,
            5869,
            5877,
            5886,
            5942,
            5994,
            6030,
            6125,
            6148,
            6171,
            6200,
            6214,
            6215,
            6216,
            6217,
            6219,
            6230,
            6238,
            6240,
            6243,
            6245,
            6248,
            6250,
            6252,
            6253,
            6257,
            6258,
            6260,
            6265,
            6267,
            6272,
            6300,
            6335,
            6388,
            6410,
            6420,
            6451,
            6461,
            6465,
            6495,
            6501,
            6506,
            6517,
            6533,
            6555,
            6621,
            6654,
            6683,
            6697,
            6753,
            6802,
            6805,
            6809,
            6908,
            6941,
            6945,
            6989,
            6995,
            7002,
            7010,
            7027,
            7028,
            7029,
            7049,
            7054,
            7056,
            7057,
            7069,
            7082,
            7086,
            7122,
            7166,
            7173,
            7175,
            7181,
            7183,
            7202,
            7208,
            7223,
            7225,
            7231,
            7232,
            7238,
            7240,
            7244,
            7245,
            7247,
            7259,
            7261,
            7275,
            7288,
            7292,
            7295,
            7308,
            7310,
            7311,
            7312,
            7313,
            7320,
            7324,
            7340,
            7341,
            7343,
            7352,
            7359,
            7385,
            7386,
            7408,
            7410,
            7413,
            7418,
            7421,
            7423,
            7430,
            7431,
            7438,
            7440,
            7443,
            7445,
            7448,
            7450,
            7451,
            7454,
            7457,
            7458,
            7460,
            7462,
            7463,
            7464,
            7468,
            7472,
            7480,
            7487,
            7488,
            7491,
            7495,
            7507,
            7525,
            7535,
            7546,
            7566,
            7585,
            7588,
            7590,
            7607,
            7615,
            7617,
            7622,
            7623,
            7651,
            7653,
            7655,
            7656,
            7663,
            7665,
            7667,
            7676,
            7689,
            7692,
            7704,
            7708,
            7711,
            7742,
            7760,
            7765,
            7782,
            7883,
            7889,
            7933,
            7966,
            7973,
            7976,
            8010,
            8013,
            8027,
            8029,
            8050,
            8062,
            8118,
            8128,
            8132,
            8199,
            8225,
            8229,
            8231,
            8322,
            8334,
            8403,
            8438,
            8446,
            8493,
            8501,
            8528,
            8532,
            8537,
            8538,
            8563,
            8564,
            8568,
            8569,
            8570,
            8571,
            8589,
            8624,
            8635,
            8659,
            8678,
            8698,
            8703,
            8710,
            8716,
            8744,
            8795,
            8805,
            8808,
            8857,
            8858,
            8890,
            8900,
            8927,
            8928,
            8958,
            8996,
            9006,
            9026,
            9031,
            9037,
            9041,
            9056,
            9061,
            9069,
            9109,
            9110,
            9120,
            9221,
            9236,
            9238,
            9269,
            9270,
            9275,
            9279,
            9285,
            9315,
            9324,
            9345,
            9355,
            9358,
            9382,
            9408,
            9567,
            9588,
            9599,
            9641,
            9642,
            9706,
            9711,
            9716,
            9717,
            9749,
            9808,
            9856,
            9857,
            9903,
            9922,
            9926,
            9942,
            9955,
            9960,
            9963,
            9981,
            9999,
            10009,
            10078,
            10082,
            10114,
            10153,
            5,
            23,
            137,
            682,
            755,
            902,
            938,
            1009,
            1012,
            1023,
            1075,
            1111,
            1143,
            1162,
            1163,
            1612,
            1704,
            1707,
            1789,
            1830,
            1866,
            1903,
            1907,
            1913,
            1933,
            2023,
            2030,
            2037,
            2078,
            2085,
            2088,
            2187,
            2208,
            2213,
            2223,
            2229,
            2231,
            2352,
            2392,
            2409,
            2427,
            2506,
            2621,
            2622,
            2679,
            3007,
            3294,
            3338,
            3389,
            3425,
            3449,
            3455,
            3459,
            3483,
            3547,
            3611,
            3809,
            3961,
            3980,
            4005,
            4044,
            4067,
            4130,
            4137,
            4228,
            4229,
            4238,
            4249,
            4293,
            4320,
            4397,
            4434,
            4448,
            4481,
            4497,
            4503,
            4528,
            4566,
            4620,
            4773,
            4881,
            5469,
            5507,
            5614,
            5697,
            5821,
            5894,
            6133,
            6213,
            6221,
            6262,
            6264,
            6269,
            6293,
            6346,
            6351,
            6384,
            6483,
            6538,
            6542,
            6597,
            6673,
            6675,
            6758,
            6815,

            ]
def compute_box_3d(dim_changed, location_changed, rotation_y):
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    rot_inds = order_rotation_y_matrix(R)
    rot_mat = R[rot_inds, :]
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

def RotationMatrixToeulerAngles(Rtilt):
    theta_x = math.atan2(Rtilt[2][1], Rtilt[2][2])
    theta_y = math.atan2(-Rtilt[2][0], np.sqrt(Rtilt[2][1]**2 + Rtilt[2][2]**2))
    theta_z = math.atan2(Rtilt[1][0], Rtilt[0][0])
    print("theta")
    print(str(theta_x*180.0/math.pi)[:6])
    print(str(theta_y*180.0/math.pi)[:6])
    print(str(theta_z*180.0/math.pi)[:6])
    theta = [float(theta_x), float(theta_y), float(theta_z)]
    return theta

def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))
    # R = R_x
    return R
def cal_rot_x(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    return R_x
def cal_rot_y(theta):
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
    return R_y
def cal_rot_z(theta):
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    return R_z
def compute_box_3d_world(dim, location, theta_c, theta):
  # print("roatation_y:" +str(rotation_y))
  c, s = np.cos(theta_c), np.sin(theta_c)
  print("rotation_y:" + str(theta_c) + "각도는" + str(theta_c * 180 / math.pi))

  R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

  # R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  print("before_np.argsort" + str(R))
  print("R[:,0]:" + str(R[:, 0]))
  print("-abs:" + str(-abs(R[:, 0])))
  print("argsort:" + str(np.argsort(-abs(R[:, 0]))))
  # rot_inds = np.argsort(-abs(R[:, 0]))

  # when R matrix가 y중심일 때
  # rot_inds = order_rotation_y_matrix(R)
  # rot_mat = R[rot_inds, :]

  #when R matrix가 z중심일 때
  rot_inds = np.argsort(-abs(R[0, :]))
  R =R[:, rot_inds]

  print("after_np.argsort" + str(R))
  rot_coeffs = abs(np.array(dim))
  x = rot_coeffs[0]
  y = rot_coeffs[1]
  z = rot_coeffs[2]
  x_corners = [-x, x, x, -x, -x, x, x, -x]
  y_corners = [y, y, -y, -y, y, y, -y, -y]
  z_corners = [z, z, z, z, -z, -z, -z, -z]
  corners2 = np.array([x_corners, y_corners, z_corners], dtype=np.float32)

  corners_3d = np.dot(R, corners2)  #(3,8)
  temp = np.array(location, dtype=np.float32).reshape(3, 1)
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3,1) #(3,8)
  corners_3d = corners_3d.transpose(1, 0)
  Rtilt_from_theta = eulerAnglesToRotationMatrix(theta)
  corners_3d = np.dot(np.transpose(Rtilt_from_theta), np.transpose(corners_3d[:, 0:3]))
  #맞나 확인
  # rot_x = cal_rot_x(theta)
  # rot_y = cal_rot_y(theta)
  # rot_z = cal_rot_z(theta)
  corners_3d = corners_3d.transpose(1, 0)
  sun_permutation = [0,2,1]
  index = np.argsort(sun_permutation)

  corners_3d = corners_3d[:, index]
  corners_3d[:, 1] *= -1
  return corners_3d
ID = {
    "bathtub" : 0,
    "bed" : 1,
    "bookshelf" :2,
    "chair":3,
    "desk":4,
    "dresser":5,
    "nightstand":6,
    "sofa":7,
    "table":8,
    "toilet":9,
}

cat_ids = {cat: i + 1 for i, cat in
           enumerate(ID)}  # cat_ids는 dict형. keyword로 cat_name : value로 번호 1~8까지가 object. 9가 don't care
# cat_info = [{"name": "pedestrian", "id": 1}, {"name": "vehicle", "id": 2}]
F = 721
H = 384  # 375
W = 1248  # 1242
EXT = [45.75, -0.34, 0.005]
CALIB = np.array([[F, 0, W / 2, EXT[0]], [0, F, H / 2, EXT[1]],
                  [0, 0, 1, EXT[2]]], dtype=np.float32)


# list형. 원소들은 dict형 - name: cat명, id: 그 cat명의 index
cat_info = []
for i, cat in enumerate(ID):
    cat_info.append({'name': cat, 'id': i + 1})
print(cat_info)
# network에 맞는 dir를 찾아가도록 추가해줌.

# image_set_path = '../../data/kitti/' + 'ImageSets_{}/'.format(SPLIT)
ann_dir = DATA_PATH + 'label_mini_new/'
# calib_dir = DATA_PATH + '{}/calib/'  # 오류인듯: 뒤에 .format(SPLIT)이 붙어야 할듯
calib_dir = DATA_PATH + 'calibIndexing/'  # 수정
Rtilt_dir = DATA_PATH + 'RtiltIndexing/'
img_dir = DATA_PATH + 'imgIndexing_backup/'
# splits = ['train', 'val']
#splits = ['testDataIndex']
# splits=['train']
splits = ['allDataIndex_ut']
# splits = ['train_untilted_ut', 'val_untilted_ut'] #사용할 때는 untitled에서 선별된 labelset과 동일하게
# splits = ['trainval', 'test']
calib_type = {'train': 'training', 'val': 'training', 'trainval': 'training',
              'test': 'testing'}
# imgIndex_list = read_imgIndex(DATA_PATH + 'train.txt')
imgIndex_list = [50,
51,
55,
115,
116,
123,
125,
155,
246,
285,
306,
388,
409,
679,
715,
716,
739,
778,
802,
818,
852,
918,
976,
1115,
1131,
1354,
1365,
1677,
1679,
1693,
1694,
1695,
1696,
1699,
1705,
1861,
1879,
1882,
1884,
1898,
1900,
1915,
1948,
2042,
2060,
2072,
2087,
2092,
2093,
2112,
2113,
2130,
2141,
2176,
2180,
2182,
2186,
2188,
2194,
2212,
2214,
2230,
2232,
2237,
2238,
2240,
2303,
2304,
2342,
2343,
2393,
2418,
2428,
2449,
2498,
2509,
2619,
2802,
2852,
2871,
2919,
2933,
2977,
3133,
3223,
3265,
3283,
3284,
3300,
3301,
3323,
3335,
3432,
3441,
3456,
3487,
3488,
3490,
3493,
3497,
3514,
3543,
3545,
3614,
3615,
3625,
3700,
3796,
3808,
3902,
3970,
4004,
4017,
4019,
4026,
4037,
4047,
4052,
4053,
4112,
4113,
4138,
4167,
4195,
4219,
4254,
4266,
4289,
4312,
4376,
4378,
4439,
4522,
4564,
4628,
4639,
4642,
4699,
4847,
4924,
5022,
5090,
5091,
5096,
5102,
5120,
5204,
5216,
5530,
5357,
5377,
5449,
5756,
5808,
5809,
5864,
5869,
5877,
5886,
5942,
5994,
6030,
6125,
6148,
6171,
6200,
6214,
6215,
6216,
6217,
6219,
6230,
6238,
6240,
6243,
6245,
6248,
6250,
6252,
6253,
6257,
6258,
6260,
6265,
6267,
6272,
6300,
6335,
6388,
6410,
6420,
6451,
6461,
6465,
6495,
6501,
6506,
6517,
6533,
6555,
6621,
6654,
6683,
6697,
6753,
6802,
6805,
6809,
6908,
6941,
6945,
6989,
6995,
7002,
7010,
7027,
7028,
7029,
7049,
7054,
7056,
7057,
7069,
7082,
7086,
7122,
7166,
7173,
7175,
7181,
7183,
7202,
7208,
7223,
7225,
7231,
7232,
7238,
7240,
7244,
7245,
7247,
7259,
7261,
7275,
7288,
7292,
7295,
7308,
7310,
7311,
7312,
7313,
7320,
7324,
7340,
7341,
7343,
7352,
7359,
7385,
7386,
7408,
7410,
7413,
7418,
7421,
7423,
7430,
7431,
7438,
7440,
7443,
7445,
7448,
7450,
7451,
7454,
7457,
7458,
7460,
7462,
7463,
7464,
7468,
7472,
7480,
7487,
7488,
7491,
7495,
7507,
7525,
7535,
7546,
7566,
7585,
7588,
7590,
7607,
7615,
7617,
7622,
7623,
7651,
7653,
7655,
7656,
7663,
7665,
7667,
7676,
7689,
7692,
7704,
7708,
7711,
7742,
7760,
7765,
7782,
7883,
7889,
7933,
7966,
7973,
7976,
8010,
8013,
8027,
8029,
8050,
8062,
8118,
8128,
8132,
8199,
8225,
8229,
8231,
8322,
8334,
8403,
8438,
8446,
8493,
8501,
8528,
8532,
8537,
8538,
8563,
8564,
8568,
8569,
8570,
8571,
8589,
8624,
8635,
8659,
8678,
8698,
8703,
8710,
8716,
8744,
8795,
8805,
8808,
8857,
8858,
8890,
8900,
8927,
8928,
8958,
8996,
9006,
9026,
9031,
9037,
9041,
9056,
9061,
9069,
9109,
9110,
9120,
9221,
9236,
9238,
9269,
9270,
9275,
9279,
9285,
9315,
9324,
9345,
9355,
9358,
9382,
9408,
9567,
9588,
9599,
9641,
9642,
9706,
9711,
9716,
9717,
9749,
9808,
9856,
9857,
9903,
9922,
9926,
9942,
9955,
9960,
9963,
9981,
9999,
10009,
10078,
10082,
10114,
10153,
5,
23,
137,
682,
755,
902,
938,
1009,
1012,
1023,
1075,
1111,
1143,
1162,
1163,
1612,
1704,
1707,
1789,
1830,
1866,
1903,
1907,
1913,
1933,
2023,
2030,
2037,
2078,
2085,
2088,
2187,
2208,
2213,
2223,
2229,
2231,
2352,
2392,
2409,
2427,
2506,
2621,
2622,
2679,
3007,
3294,
3338,
3389,
3425,
3449,
3455,
3459,
3483,
3547,
3611,
3809,
3961,
3980,
4005,
4044,
4067,
4130,
4137,
4228,
4229,
4238,
4249,
4293,
4320,
4397,
4434,
4448,
4481,
4497,
4503,
4528,
4566,
4620,
4773,
4881,
5469,
5507,
5614,
5697,
5821,
5894,
6133,
6213,
6221,
6262,
6264,
6269,
6293,
6346,
6351,
6384,
6483,
6538,
6542,
6597,
6673,
6675,
6758,
6815,
]

for split in splits:
    ret = {'images': [], 'annotations': [], "categories": cat_info}
    image_set = open(DATA_PATH + '{}.txt'.format(split), 'r')
    #skip image확인하기위해서
    # image_set = imgIndex_list
    # image_set = list(range(10335))
    image_to_id = {}
    for line in image_set:  # train.txt에 있는 line을 str로 하나씩 읽어들임
        # if line[-1] == '\n':  # line(str)의 마지막에 \n있으면 빼주고.
        #     line = line[1:-2]
        #     print(line)
        # image_id = line[0] # ex.000003 → 3
        # image_path = line[1][1:-1]

        # image_name = image_path.split('/')[-1]
        # calib_path = calib_dir.format(calib_type[split]) + '{}.txt'.format(image_path)

        image_id = line[:-1]
        # image_id = 6
        # print(image_id)

        # image_id = line
        #원하는 이미지 id  사용시
        # image_id = 5
        # if image_id in skip_img:
        #     continue
        # if (image_id == '89'):
        #     print(line)
        img_path = img_dir + '{}.jpg'.format(image_id)
        calib_path = calib_dir + '{}.txt'.format(image_id)
        Rtilt_path = Rtilt_dir + '{}.txt'.format(image_id)

        calib = read_clib(calib_path)
        Rtilt = read_Rtilt(Rtilt_path)
        print(Rtilt)
        Rtilt_aug = Rtilt.flatten().tolist()
        f = open('../../data/SUNRGBD/Rtilt_Aug/' + '{}.txt'.format(int(image_id) + 10335), 'w')
        f.write(str(Rtilt_aug))
        Rtilt_aug.clear()
        image_info = {'file_name': img_path.split('/')[-1],
                      'id': int(image_id),
                      'calib': calib.tolist()}  # 1자로 펴서 넣어준다.
        ret['images'].append(image_info)
        if split == 'test':
            continue
        ann_path = ann_dir + '{}.txt'.format(image_id)  # line = train.txt = image의 번호
        # if split == 'val':
        #   os.system('cp {} {}/'.format(ann_path, VAL_PATH))
        anns = open(ann_path, 'r')

        if DEBUG:
            # image = cv2.imread(image_info['file_name'])
            image = cv2.imread(img_path)
            # cv2.imwrite('C:\\obj_detection\\CenterNet-master\\CenterNet-master\\data\\SUNRGBD\\img_Aug\\' + str(
            #     int(image_id) + 10335) + '.jpg', image)

        label_lists =[]
        for ann_ind, txt in enumerate(anns):  # 한줄씩 처리한다는 의미: ann_ind는 object 갯수에 대한 index.
            tmp = txt[1:-2].split(',')  # white space제거
            # cat_index = cat_ids[tmp[0][1:-1]]  # 가장 앞에 있는 str은 object의 이름
            cat_id = tmp[0][1:-1]
            '''
            not be used
            truncated = int(
                float(tmp[1]))  # truncated refers to the object leaving image boundaries. 화면에 남아있는 정도를 의미
            occluded = int(
                tmp[2])
            alpha = float(tmp[3])  # Observation angle of object, ranging [-Pi; Pi]
            bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(
                                tmp[7])]  # (0-based) bounding box of the object: Left, top, right, bottom image coordinates

            '''
            # 가려진 정도를 의미 0 = fully visible, 1 = partly occluded 2 = largely occluded, 3 = unknown
            # tmp순서: centroid_x, centroid_y, centroid_z, length, height, width
            # 방법1) x,z,y수넛로 값을 집어 넣기
            location = [float(tmp[1]), float(tmp[2]), float(tmp[3])]
            # 3D object location x,z,y in camera coords. [m]
            # 왜냐하면 SUN-RGBD는 x,z평면을 이용하기 때문에, location의 (x,y)위치에 SUN-RGBD의 (x,z)가 들어가줘야한다.
            # dim = [float(tmp[5]), float(tmp[6]), float(tmp[4])]  # 3D object dimensions: height, width, length [m]
            dim = [float(tmp[4]), float(tmp[5]), float(tmp[6])]  # 3D object dimensions: x길이, y길이, z길이
            rotation_y = float(tmp[7])  # Rotation around Y-axis in camera coords. [-Pi; Pi]
            bbox2D = [float(tmp[8]), float(tmp[9]), float(tmp[10]), float(tmp[11])]# (0-based) bounding box of the object: Left, top, right, bottom image coordinates
            xmin = int(bbox2D[0])
            ymin = int(bbox2D[1])
            xmax = int(bbox2D[0]) + int(bbox2D[2])
            ymax = int(bbox2D[1]) + int(bbox2D[3])
            cv2.line(image, (xmin, ymin), (xmax, ymin), (0, 255, 0), 3, lineType=cv2.LINE_AA)
            cv2.line(image, (xmax, ymin), (xmax, ymax), (0, 255, 0), 3, lineType=cv2.LINE_AA)
            cv2.line(image, (xmin, ymax), (xmin, ymin), (0, 255, 0), 3, lineType=cv2.LINE_AA)
            cv2.line(image, (xmax, ymax), (xmin, ymax), (0, 255, 0), 3, lineType=cv2.LINE_AA)

            ct_x = xmin + bbox2D[2] / 2
            ct_y = ymin + bbox2D[3] / 2
            cv2.circle(image, (int(ct_x), int(ct_y)), 10, (0, 255, 0), -1)

            #alpha = rot_y2alpha(rotation_y, x, calib[0, 2], calib[0, 0])

 #파랑

            # rotation_y = -rotation_y
            theta = RotationMatrixToeulerAngles(Rtilt)
            box_3d = compute_box_3d_world(dim, location, rotation_y, theta)

            # print(cat_id, location, dim, rotation_y, bbox2D, alpha)
            # ann에 axis 바꾼거를 집어넣기
            sun_permutation = [0, 2, 1]
            #19.12.26 추가된 부분
            location = np.dot(np.transpose(Rtilt), location)

            location_changed = [location[i] for i in sun_permutation]
            location_changed[1] *= -1

            #dim_changed = [dim[i]*2 if dim[i] > 0 else print("error" + str(image_id)) for i in sun_permutation]  #왜냐하면 dimension은 크기니까
            dim_changed = [dim[i]*2 for i in sun_permutation]  #왜냐하면 dimension은 크기니까
            dim_xyz = [dim[i]*2 for i in range(0,3)]
            #Rtilt 적용된 location을 2d image에 project시킨것.
            ct3d_to_ct2d = project_to_image(np.array([location_changed]), calib)  # 이제 3d bounding box를 image에 투영시킴
            cv2.circle(image, (int(ct3d_to_ct2d[0][0]), int(ct3d_to_ct2d[0][1])), 10, (0, 0, 255), -1)
            alpha = rotation_y - np.arctan2(ct3d_to_ct2d[0][0] - calib[0][2], calib[0][0])
            if alpha > np.pi:
                alpha -= 2 * np.pi
            if alpha < -np.pi:
                alpha += 2 * np.pi
            alpha_x = ct3d_to_ct2d[0][0] + np.cos(alpha)*100
            alpha_y = ct3d_to_ct2d[0][1] - np.sin(alpha)*100
            cv2.arrowedLine(image, (int(ct3d_to_ct2d[0][0]), int(ct3d_to_ct2d[0][1])), (int(alpha_x), int(alpha_y)), (255,0,0), 2)

            end_x = ct3d_to_ct2d[0][0] + np.cos(rotation_y) * 100
            end_y = ct3d_to_ct2d[0][1] - np.sin(rotation_y) * 100
            cv2.arrowedLine(image, (int(ct3d_to_ct2d[0][0]), int(ct3d_to_ct2d[0][1])), (int(end_x), int(end_y)), (0, 255, 0), 2)
            # box_3d = compute_box_3d(dim_changed, location_changed, rotation_y)
            # print("before neg")
            # print(rotation_y)
            # rotation_y = -rotation_y
            # print("after neg")
            # print(rotation_y)
            # 만약 rotation_y의 좌표를 반대로 넣고 싶을 때


            #-rotation으로 돌리려는 상황

            # print(dim, location, rotation_y, Rtilt)
            # print(box_3d)
            box_2d = project_to_image(box_3d, calib)  # 이제 3d bounding box를 image에 투영시킴
            #print(box_2d.transpose()[0])
            #print(box_2d.transpose()[1])
            xmin_projected = box_2d.transpose()[0].min()
            xmax_projected = box_2d.transpose()[0].max()
            ymin_projected = box_2d.transpose()[1].min()
            ymax_projected = box_2d.transpose()[1].max()


            height, width, channel = image.shape #(530, 730)
            if xmin_projected < 0 :
                xmin_projected = 0
            if ymin_projected < 0 :
                ymin_projected = 0
            if xmax_projected >= width:
                xmax_projected = width - 1
            if ymax_projected >= height:
                ymax_projected = height -1

            ct_x_projected= (xmin_projected + xmax_projected) /2
            ct_y_projected = (ymin_projected + ymax_projected) /2
            # location_changed를 project시킨 값이 image 크기 내에 존재할 때
            label_list = []
            if(0<=ct3d_to_ct2d[0][0]<width) and (0 <=ct3d_to_ct2d[0][1]< height):
                label_list = txt[:-1]
                # label_list = [cat_id, location[0], location[1], location[2], dim[0], dim[1], dim[2], rotation_y,
                #               bbox2D[0], bbox2D[1], bbox2D[2], bbox2D[3], alpha]
                label_lists.append(label_list)
        if len(label_lists) == 0:
            continue
        fp = open('../../data/SUNRGBD/label_Aug/' + '{}.txt'.format(int(image_id)+ 10335), 'w')
        for j in range(len(label_lists)):
            fp.write(str(label_lists[j]) + '\n')

        fp.close()
        label_lists.clear()
        # cv2.imshow(str(image_id), image)
        # cv2.waitKey()
            #
            #     width_projected = xmax_projected - xmin_projected
            #     height_projected = ymax_projected - ymin_projected
            #
            #     # print(height, width, channel)
            #
            #     #alpha를 projectd 3d center 기준으로 구할 때
            #     # alpha = rotation_y - np.arctan2(ct_x_projected - calib[0][2], calib[0][0])
            #     # if alpha > np.pi:
            #     #     alpha -= 2 * np.pi
            #     # if alpha < -np.pi:
            #     #     alpha += 2 * np.pi
            #     if (location_changed[2] < 0):
            #         print("minus location_z")
            #         print(location_changed[2])
            #     #draw yellow(projected 3d bounding box)
            #     # cv2.line(image, (int(xmin_projected), int(ymin_projected)), (int(xmax_projected), int(ymin_projected)), (0, 255, 255), 2, lineType=cv2.LINE_AA)
            #     # cv2.line(image, (int(xmax_projected), int(ymin_projected)), (int(xmax_projected), int(ymax_projected)), (0, 255, 255), 2, lineType=cv2.LINE_AA)
            #     # cv2.line(image, (int(xmin_projected), int(ymax_projected)), (int(xmin_projected), int(ymin_projected)), (0, 255, 255), 2, lineType=cv2.LINE_AA)
            #     # cv2.line(image, (int(xmax_projected), int(ymax_projected)), (int(xmin_projected), int(ymax_projected)), (0, 255, 255), 2, lineType=cv2.LINE_AA)
            #     #
            #     # cv2.circle(image, (int(ct_x_projected),int(ct_y_projected)), 5, (0,255,255), -1)
            #
            #     #print(box_2d.transpose()[0].min(), box_2d.transpose()[0].max(), box_2d.transpose()[1].min(), box_2d.transpose()[1].max())
            #     ct3d_proj = [float(ct3d_to_ct2d[0][0]), float(ct3d_to_ct2d[0][1])]
            #     #bbox2D_projected는 (xmin, ymin, width, height꼴로 넣어줘야함)
            #     bbox2D = [float(bbox2D[0]), float(bbox2D[1]), float(bbox2D[2]), float(bbox2D[3])]
            #     ann = {'image_id': int(image_id),
            #            'id': int(len(ret['annotations']) + 1),
            #            'category_id': cat_ids[cat_id],
            #            'dim': dim_xyz,
            #            'bbox': bbox2D,#(xmin, ymin, w, h)
            #            #'depth': abs(location_changed[1]), ....NotRtilted(2019-12-23)
            #            'depth': location_changed[2],
            #            'alpha': alpha,
            #            # 'truncated': truncated,
            #            # 'occluded': occluded,
            #            'location': location_changed,
            #            'rotation_y': rotation_y
            #            }
            #
            #     ret['annotations'].append(ann)
            #     image = draw_box_3d_world(image, box_2d, image_id)

            #     # cv2.imwrite('C:\\obj_detection\\CenterNet-master\\CenterNet-master\\data\\SUNRGBD\\resultImages_mini\\'+str(image_id)+'.jpg',image)
            #
            # else:
            #     print("3d_ct out")
            #     print(image_id)
            #     print(cat_id)



            # depth = np.array([location[2]], dtype=np.float32)
            # pt_2d = np.array([(bbox2D[0] + bbox2D[2]) / 2, (bbox2D[1] + bbox2D[3]) / 2],
            #                  dtype=np.float32)
            # pt_3d = unproject_2d_to_3d(pt_2d, depth, calib)
            # pt_3d[1] += dim[0] / 2  # because the position of  KITTI is defined as the center of the bottom face
            # print('pt_3d', pt_3d)
            # print('location', location)


    # 이미지 보이기
    # if DEBUG:

    # print("# images: ", len(ret['images']))
    # print("# annotations: ", len(ret['annotations']))
    # # # # # import pdb; pdb.set_trace()
    # out_path = '{}/annotations/sun_{}.json'.format(DATA_PATH, split)
    # json.dump(ret, open(out_path, 'w'))
