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
import _init_paths
from utils.ddd_utils import project_to_image, project_to_image_sun2, project_to_image_sun, alpha2rot_y, \
    compute_box_3d_sun, compute_box_3d_sun_2, compute_box_3d_sun_3, compute_box_3d_sun_4, compute_box_3d_sun_5, \
    compute_box_3d_sun_6, compute_box_3d_sun_8, compute_box_3d_sun_10
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d, draw_box_3d_sun
from utils.ddd_utils import rot_y2alpha, order_rotation_y_matrix, draw_box_3d_world
from utils.image import get_affine_transform, affine_transform, get_3rd_point, get_dir
DATA_PATH = '../../data/SUNRGBD/'


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

def removeWhiteSpaceInfile(f_lines):

    f_list = []
    for line in f_lines:
        if line[-1] == '\n':
            f_list.append(line[:-1])
        else:
            print("no whiteline")
            f_list.append(line)

    return f_list


def read_gt_label(gt_label):
    cat_id = gt_label[0][2:-1]
    loc_gt_w = [float(tmp[1]), float(tmp[2]), float(tmp[3])]
    dim_gt_w= [float(tmp[4]), float(tmp[5]), float(tmp[6])]  # 3D object dimensions: x길이, y길이, z길이
    rot_gt_w = float(tmp[7])  # Rotation around Y-axis in camera coords. [-Pi; Pi]
    bbox2D_gt = [float(tmp[8]), float(tmp[9]), float(tmp[10]), float(
        tmp[11])]
    return cat_id, [loc_gt_w, dim_gt_w, rot_gt_w, bbox2D_gt]


def read_pd_label(gt_label):
    cat_id = gt_label[0][1:-1]
    loc_pd_c = [float(gt_label[1]), float(gt_label[2]), float(gt_label[3])]
    dim_pd_w = [float(gt_label[4]), float(gt_label[5]), float(gt_label[6])]  # 3D object dimensions: x길이, y길이, z길이
    rot_pd_w = float(gt_label[7][:-1])  # Rotation around Y-axis in camera coords. [-Pi; Pi]
    return cat_id, [loc_pd_c, dim_pd_w, rot_pd_w]
def get_iou2d(bb1, bb2):
    #bb1, bb2는 list형
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[1])
    y_bottom = min(bb1[2], bb2[2])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def draw_2d_box(color_raw, xmin, ymin, xmax, ymax, c=(0, 255, 0)):
    cv2.line(color_raw, (int(xmin), int(ymin)), (int(xmax), int(ymin)), c, 2, lineType=cv2.LINE_AA)
    cv2.line(color_raw, (int(xmax), int(ymin)), (int(xmax), int(ymax)), c, 2, lineType=cv2.LINE_AA)
    cv2.line(color_raw, (int(xmax), int(ymax)), (int(xmin), int(ymax)), c, 2, lineType=cv2.LINE_AA)
    cv2.line(color_raw, (int(xmin), int(ymax)), (int(xmin), int(ymin)), c, 2, lineType=cv2.LINE_AA)
    ct_x = (xmin + xmax) / 2
    ct_y = (ymin + ymax) / 2
    cv2.circle(color_raw, (int(ct_x), int(ct_y)), 3, c, 2, lineType=cv2.LINE_AA)
'''
untilt의 경우, 저장될 때, 
label,
center_cam (x,y,z)
dim_world 
theta
----------------------
x_min, y_min, x_max, y_max
 '''
# cat_ids는 dict형. keyword로 cat_name : value로 번호 1~8까지가 object.

ID = {
    "bathtub": 0,
    "bed": 1,
    "bookshelf": 2,
    "chair": 3,
    "desk": 4,
    "dresser": 5,
    "nightstand": 6,
    "sofa": 7,
    "table": 8,
    "toilet": 9,
}

#cat_ids : cat_name의 number에 +1 해서 dict형을 만든 상황
#i는 for문의 index cat은 keyword이고,
cat_ids = {cat: i + 1 for i, cat in enumerate(ID)}
print(cat_ids)

#list 내에 각 원소들이 dict형인데, 그 dict형은 2개의 keyword (name, id)를 가지고 있음.
cat_info = []
for i, cat in enumerate(ID):
    cat_info.append({'name': cat, 'id': i })


splits = ['val_untilted_ut']
ann_dir = DATA_PATH + 'label_iou03/'
calib_dir = DATA_PATH + 'calib_untilt/'
Rtilt_dir = DATA_PATH + 'RtiltIndexing/'
img_dir = DATA_PATH + 'img_backwarping_i_all/'

# pd2_dir = DATA_PATH + 'label_pd_iou03_2d3/'
# pd3_dir = DATA_PATH + 'label_pd_iou03_3d3/'
s = np.array([640, 480], dtype = np.int32)
c = np.array([320, 240], dtype = np.int32)
for split in splits:
    f = open(DATA_PATH + '{}.txt'.format(split), 'r')
    f_lines = f.readlines()
    f_list = removeWhiteSpaceInfile(f_lines)
    # image_label들에 대해서 (for split문)
    for image_id in f_list:
        print(image_id)
        img_path = img_dir + '{}.jpg'.format(image_id)
        calib_path = calib_dir + '{}.txt'.format(image_id)
        Rtilt_path = Rtilt_dir + '{}.txt'.format(image_id)
        ann_path = ann_dir + '{}.txt'.format(image_id)
        img = cv2.imread(img_path)
        img_w, img_h = img.shape[1], img.shape[0]

        gt_dict_per_img = {"bathtub": [], "bed": [], "bookshelf": [],
                           "chair": [], "desk": [], "dresser": [], "nightstand": [],
                           "sofa": [], "table": [], "toilet": []}
        pd_dict_per_img = {"bathtub": [], "bed": [], "bookshelf": [],
                           "chair": [], "desk": [], "dresser": [], "nightstand": [],
                           "sofa": [], "table": [], "toilet": []}
        anns = open(ann_path, 'r')
        anns_lines = anns.readlines()
        anns_list = removeWhiteSpaceInfile(anns_lines)
        # generate gt_dict_per_img
        for txt in anns_list:
            tmp = txt.split(',')  # white space제거
            cat_id, gt_label_list = read_gt_label(tmp)
            gt_dict_per_img[cat_id].append(gt_label_list)

        # pd_2d
        pd2_path = pd2_dir + '{}.txt'.format(image_id)
        pd3_path = pd3_dir + '{}.txt'.format(image_id)
        pd2 = open(pd2_path, 'r')

        pd2_lines = pd2.readlines()
        if len(pd2_lines) == 0:
            print("empty gt file")
            continue
        pd2_list = removeWhiteSpaceInfile(pd2_lines)

        pd3 = open(pd3_path, 'r')
        pd3_lines = pd3.readlines()
        pd3_list = removeWhiteSpaceInfile(pd3_lines)

        # generate gt_dict_per_img
        for txt_3d in pd3_list:
            tmp_3d = txt_3d[1:-1].split(',')
            cat_3d, pd_label_list = read_pd_label(tmp_3d)
            pd_dict_per_img[cat_3d].append(pd_label_list)

        # pd2_bbox per object
        # for ann_ind, txt in enumerate(pd2):

        for txt in pd2_list:
            tmp = txt[1:-3].split(',')
            cat_2d = tmp[0][1:-1]
            score_2d = float(tmp[1])
            bbox2D = np.array([float(tmp[2][2:]), float(tmp[3]), float(tmp[4]), float(tmp[5])], dtype = np.float32)

            trans_box = get_affine_transform(c, s, 0, [img_w, img_h])
            bbox2D[:2] = affine_transform(bbox2D[:2], trans_box)  # (112, 112)에 맞게 BR 점의 위치를 옮겨줌
            bbox2D[2:] = affine_transform(bbox2D[2:], trans_box)
            # #np.clip함수를 사용하여 bbox행렬의 범위를 output_w, output-h 내의 범위로 바꿔줌

            bbox2D[[0, 2]] = np.clip(bbox2D[[0, 2]], 0, img_w - 1)
            bbox2D[[1, 3]] = np.clip(bbox2D[[1, 3]], 0, img_h - 1)
            #draw_2d_pd
            draw_2d_box(img, bbox2D[0], bbox2D[1], bbox2D[2], bbox2D[3], c=(0, 255, 0))

            # cv2.imshow("img", img)
            # cv2.waitKey()

            # 하나의 pd_2d에 대해서 그것과 동일한 cat에 대해서 all gt_2d
            key_label_lists = gt_dict_per_img[cat_2d]
            # for key in gt_dict_per_img :
            #     if cat_2d == key:
            #         key_label_lists =gt_dict_per_img[key]
            # 그 all gt_2d와 하나의 pd_2d iou를 모두 비교하여 iou_2d_list에 저장함.
            iou_2d_list = []
            # pd_2d가 찾은 cat의 gt_dict가 비어있는 상황.
            if len(key_label_lists) == 0:
                continue

            for key_label_list in key_label_lists:
                key_bbox2d = key_label_list[3]
                iou_2d = get_iou2d(bbox2D.tolist(), key_bbox2d)
                iou_2d_list.append(iou_2d)
            # print("iou_2d_list" + str(len(iou_2d_list)))

            assert len(key_label_lists) == len(iou_2d_list)
            # print("iou_2d_list")
            # print(iou_2d_list)
            max_iou = max(iou_2d_list)
            #동일한 cat임은 인정. 그런데 겹침이 하나도 없는 곳.
            if int(max_iou) == 0:
                continue
            key_index = iou_2d_list.index(max_iou)
            bbox2D_gt = key_label_lists[key_index][3]
            draw_2d_box(img, bbox2D_gt[0], bbox2D_gt[1], bbox2D_gt[2], bbox2D_gt[3], c= (0, 0, 255))

            pd3_label = gt_dict_per_img[cat_2d][key_index]
            print(pd3_label)
            # for key in pd_dict_per_img:
            #     if cat_2d == key:
            #         pd3_key= gt_dict_per_img[key][key_index]

        #선별된 pd에 해당하는 key_index에 위치한 pd_3d를 가져와야함.

        # cv2.imshow("img", img)
        # cv2.waitKey()
        # gt_labels_dict = read_gt_label(ann)
        # for ann_ind, txt in enumerate(anns):  # 한줄씩 처리한다는 의미: ann_ind는 object 갯수에 대한 index.

    # train.txt에 있는 line을 str로 하나씩 읽어들임
    # gt :anns 갯수만큼 에 있는 값들 모두 list에 저장
    # pd_2d : anns 갯수만큼 모두 list에 저장
    # pd_3d: pd_2d의 순서랑 동일하게 anno 갯수를 지니고 있을 것.
