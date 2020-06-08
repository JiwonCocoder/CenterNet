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
from matplotlib import pyplot as plt

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

def compute_3d_box(dim, centroid, rot_mat):
    x = dim[0] / 2
    y = dim[1] / 2
    z = dim[2] / 2
    x_corners = [-x, x, x, -x, -x, x, x, -x]
    y_corners = [y, y, -y, -y, y, y, -y, -y]
    z_corners = [z, z, z, z, -z, -z, -z, -z]
    corners2 = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(rot_mat, corners2)  # (3,8)
    corners_3d = corners_3d + np.array(centroid, dtype=np.float32).reshape(3, 1)  # (3,8)
    corners_3d = corners_3d.transpose(1, 0)
    sun_permutation = [0, 2, 1]
    index = np.argsort(sun_permutation)
    corners_3d = corners_3d[:, index]
    corners_3d[:, 1] *= -1
    # corners_3d = np.array(corners_3d, dtype=np.int64)
    return corners_3d

def read_gt_label(gt_label):
    cat_id = gt_label[0][2:-1]
    loc_gt_w = [float(gt_label[1]), float(gt_label[2]), float(gt_label[3])]
    dim_gt_w= [float(gt_label[4])*2, float(gt_label[5])*2, float(gt_label[6])*2]  # 3D object dimensions: x길이, y길이, z길이
    rot_gt_w = float(gt_label[7])  # Rotation around Y-axis in camera coords. [-Pi; Pi]
    bbox2D_gt = [float(gt_label[8]), float(gt_label[9]), float(gt_label[10]), float(gt_label[11])]
    c, s = np.cos(rot_gt_w), np.sin(rot_gt_w)
    if (-abs(c) > -abs(s)):
        theta_real = math.atan2(c, -s)  # (y/x)
        # print(theta_real, str(theta_real * 180 / math.pi)[:6])
    else:
        theta_real = rot_gt_w
    c_real, s_real = np.cos(theta_real), np.sin(theta_real)
    rot_mat = np.array([[c_real, -s_real, 0], [s_real, c_real, 0], [0, 0, 1]], dtype=np.float32)

    box_3d = compute_3d_box(dim_gt_w, loc_gt_w, rot_mat)

    return cat_id, [loc_gt_w, dim_gt_w, rot_gt_w, bbox2D_gt], box_3d


def read_pd_label(pd_label):
    cat_id = pd_label[0][1:-1]
    # loc_pd_c = [float(pd_label[1]), float(pd_label[2]), float(pd_label[3])]
    loc_pd_w = [float(pd_label[1]), float(pd_label[3]), -float(pd_label[2])]
    dim_pd_w = [float(pd_label[4]), float(pd_label[5]), float(pd_label[6])]  # 3D object dimensions: x길이, y길이, z길이
    rot_pd_w = float(pd_label[7])
    bbox2D_pd = [float(pd_label[8]), float(pd_label[9]), float(pd_label[10]), float(pd_label[11][:-1])]
    # Rotation around Y-axis in camera coords. [-Pi; Pi]
    return cat_id, [loc_pd_w, dim_pd_w, rot_pd_w, bbox2D_pd]
def get_iou2d(bb1, bb2):
    #bb1, bb2는 list형
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])
    print("pred")
    print(bb1)
    print("ground truth")
    print(bb2)
    if x_right < x_left or y_bottom < y_top:
        print("here")
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
    print(iou)
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

pd_dir = DATA_PATH + 'label_pd_untilt_2d3d3/'

s = np.array([640, 480], dtype = np.int32)
c = np.array([320, 240], dtype = np.int32)
for split in splits:
    f = open(DATA_PATH + '{}.txt'.format(split), 'r')
    f_lines = f.readlines()
    f_list = removeWhiteSpaceInfile(f_lines)
    # image_label들에 대해서 (for split문)
    error_dict_per_cat = {"bathtub": [], "bed": [], "bookshelf": [],
                          "chair": [], "desk": [], "dresser": [], "nightstand": [],
                          "sofa": [], "table": [], "toilet": []}
    error_mean_dict_per_cat = {"bathtub": [], "bed": [], "bookshelf": [],
                       "chair": [], "desk": [], "dresser": [], "nightstand": [],
                       "sofa": [], "table": [], "toilet": []}
    for image_id in f_list:

        print(image_id)
        img_path = img_dir + '{}.jpg'.format(image_id)
        calib_path = calib_dir + '{}.txt'.format(image_id)
        Rtilt_path = Rtilt_dir + '{}.txt'.format(image_id)
        ann_path = ann_dir + '{}.txt'.format(image_id)
        calib = read_clib(calib_path)
        img = cv2.imread(img_path)
        img_w, img_h = img.shape[1], img.shape[0]
        trans_box = get_affine_transform(c, s, 0, [img_w, img_h])
        gt_dict_per_img = {"bathtub": [], "bed": [], "bookshelf": [],
                           "chair": [], "desk": [], "dresser": [], "nightstand": [],
                           "sofa": [], "table": [], "toilet": []}
        pd_dict_per_img = {"bathtub": [], "bed": [], "bookshelf": [],
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
            cat_id, gt_label_list, box_3d = read_gt_label(tmp)
            box_2d = project_to_image(box_3d, calib)
            draw_box_3d(img, box_2d, (128, 128, 128))
            gt_dict_per_img[cat_id].append(gt_label_list)
            # cv2.imshow(str(image_id), img)
            # cv2.waitKey()
        # pd_2d
        pd_path = pd_dir + '{}.txt'.format(image_id)
        pd = open(pd_path, 'r')

        pd_lines = pd.readlines()
        if len(pd_lines) == 0:
            print("empty gt file")
            continue
        pd_list = removeWhiteSpaceInfile(pd_lines)

        # generate pd_dict_per_img
        for txt_3d in pd_list:
            tmp_pd = txt_3d[1:-1].split(',')
            cat_pd, pd_label_list = read_pd_label(tmp_pd)
            pd_dict_per_img[cat_pd].append(pd_label_list)

        # pd2_bbox per object
        # for ann_ind, txt in enumerate(pd2):

        for key in pd_dict_per_img:
            #하나의 cat마다 수행됨

            key_val_pd_list = pd_dict_per_img.get(key)
            key_val_gt_list = gt_dict_per_img.get(key)

            #pd는 있는데,
            if len(key_val_pd_list)> 0:
                #gt는 없는 상황
                if len(key_val_gt_list) == 0:
                    continue
                # pd가 수행되는데, 모든 동일cat의 gt에 대해서 비교해야하니까
                iou_array = np.zeros(( len(key_val_gt_list), len(key_val_pd_list)))
                gtKey_pdIOU_dict = {i : [] for i in range(len(key_val_gt_list))}
                pdKey_gtIOU_dict = {i : [] for i in range(len(key_val_pd_list))}

                for i in range(len(key_val_pd_list)):
                    print("{}번째 pd".format(i))
                    bbox2D_pd = key_val_pd_list[i][3]
                    for j in range(len(key_val_gt_list)):
                        bbox2D_gt = key_val_gt_list[j][3]
                        iou_2d = get_iou2d(bbox2D_pd, bbox2D_gt)
                        iou_array[j][i] = iou_2d
                #pd마다 gt개만큼 지니고 있는 것
                iou_list = np.transpose(iou_array).tolist()
                check_iou_overlap = []
                for i,v in enumerate(iou_list):
                    max_per_pd = max(v)
                    if max_per_pd == 0:
                        continue
                    # if max_per_pd < 0.3:
                    #     continue
                    #pd와 max_iou를 지니는 gtIndex
                    gtIndex_per_max = v.index(max_per_pd)
                    # key: gt  // val : [pd, max_iou]
                    gtKey_pdIOU_dict[gtIndex_per_max].append([i, max_per_pd])

                for gt in gtKey_pdIOU_dict :
                    # val갯수 chekc를 위해
                    pdIOU_list = gtKey_pdIOU_dict.get(gt)
                    if len(pdIOU_list) == 0:
                        continue
                    #하나의 gt에 2개이상이 detect된상황
                    if len(pdIOU_list) > 1:
                        array_iou = np.array(pdIOU_list)[:, 1]
                        iou = array_iou.max()
                        index_pd = pdIOU_list[array_iou.argmax()][0]
                    if len(pdIOU_list) == 1:
                        iou = pdIOU_list[0][1]
                        index_pd = pdIOU_list[0][0]
                    pdKey_gtIOU_dict[index_pd].append([gt, iou])

                for pd in pdKey_gtIOU_dict:
                    gtIOU_list = pdKey_gtIOU_dict.get(pd)
                    print("eval {}번째 pd".format(pd))
                    if len(gtIOU_list) == 0:
                        continue
                    if len(gtIOU_list) > 1:
                        array_iou = np.array(gtIOU_list)[:, 1]
                        iou = array_iou.max()
                        # index_gt = array_iou.argmax()
                        index_gt = gtIOU_list[array_iou.argmax()][0]
                    if len(gtIOU_list) == 1:
                        iou = gtIOU_list[0][1]
                        index_gt = gtIOU_list[0][0]
                    if iou < 0.3:
                        continue
                    #이제 pd마다 하나씩의 gt와 그에 대한 iou가 나온 상황
                    loc_w_pd_eval = np.array(key_val_pd_list[pd][0])
                    dim_w_pd_eval = np.array(key_val_pd_list[pd][1])
                    yaw_pd_eval = key_val_pd_list[pd][2]
                    bbox2D_pd_eval = key_val_pd_list[pd][3]

                    loc_w_gt_eval = np.array(key_val_gt_list[index_gt][0])
                    dim_w_gt_eval = np.array(key_val_gt_list[index_gt][1])
                    yaw_gt_eval = key_val_gt_list[index_gt][2]
                    bbox2D_gt_eval = key_val_gt_list[index_gt][3]

                    error_loc_w = (abs(loc_w_pd_eval - loc_w_gt_eval)).tolist()
                    error_dim_w = (abs(dim_w_pd_eval - dim_w_gt_eval)).tolist()
                    #yaw angle: abs 없앰
                    error_yaw = (abs(yaw_pd_eval - yaw_gt_eval))
                    # error_dict_per_cat[key].append([error_loc_w.tolist(), error_dim_w.tolist(), error_yaw])
                    error_dict_per_cat[key].append([error_loc_w[0], error_loc_w[1], error_loc_w[2], error_dim_w[0], error_dim_w[1], error_dim_w[2], error_yaw])
                    # print("bbox2D_pd_eval")
                    # print(bbox2D_pd_eval)
                    # print("bbox2D_gt_eval")
                    # print(bbox2D_gt_eval)
                    draw_2d_box(img, bbox2D_pd_eval[0], bbox2D_pd_eval[1], bbox2D_pd_eval[2], bbox2D_pd_eval[3], c=(0, 255, 0))
                    #red : gt
                    draw_2d_box(img, bbox2D_gt_eval[0], bbox2D_gt_eval[1], bbox2D_gt_eval[2], bbox2D_gt_eval[3] , c=(0, 0, 255))
                    # cv2.imshow("img", img)
                    # cv2.waitKey()
    key_error_list= []
    x_yaw = np.arange(-180,180,10)
    for key_error in error_dict_per_cat:
        if(key_error == 'nightstand'):
            continue
        error_list = error_dict_per_cat.get(key_error)
        error_list_len = len(error_list)
        error_array = np.array(error_list)
        #for histogram
        error_array[:,6] = np.rint((error_array[:,  6]* 180/ math.pi))
        # np.histogram(error_array[:,6], bins = x_yaw)
        plt.hist(error_array[:, 6], x_yaw)
        plt.title(key_error)
        plt.show()
        error_mean_array = np.mean(error_array, axis = 0)
        error_mean_array = np.round(error_mean_array, 4)
        error_mean_dict_per_cat[key_error].append(error_mean_array.tolist())
    print(error_mean_dict_per_cat)
        # key_error_list.append(error_mean_array.tolist())
                    #green: pd

                    #prediction (pd)와 ground truth(index_gt)의 3d labels





                #     check_iou_overlap.append([gtIndex_per_max, max_per_pd])
                # gt_check_list = np.array(check_iou_overlap, dtype= np.int16)[:, 0]
                # # if gt_check_list(gt_index) > 1:
                #
                # for c in range(len(check_iou_overlap)):
                #     gt_check = check_iou_overlap[c][0]
                #     iou_check = check_iou_overlap[c][1]
                #     #해당 prediction은 사용하지 않을것
                #     if gt_check == -1:
                #         continue
                #
                #     # for cc in range(1, len(check_iou_overlap)):
                #     #
                #     # if check_iou_overlap[i][0] == -1 :
                #     #     continue
                #
                #     draw_2d_box(img, bbox2D_pd_eval[0], bbox2D_pd_eval[1], bbox2D_pd_eval[2], bbox2D_pd_eval[3], c=(0, 255, 0))
                #     draw_2d_box(img, bbox2D_gt_eval[0], bbox2D_gt_eval[1], bbox2D_gt_eval[2], bbox2D_gt_eval[3] , c=(0, 0, 255))
                #     cv2.imshow("img", img)
                #     cv2.waitKey()

                # max_per_gt = iou_array.max(axis = 0)
                # print(max_per_gt)
                # if max_per_gt == 0:
                #     continue

                # max_per_index = iou_array.index(max_per_gt)
                # bbox2D_gt = key_val_gt_list[max_per_index][3]

                # for key_val_pd in key_val_pd_list:
                #     bbox2D_pd = key_val_pd[3]
                #     iou_2d_list = []
                #     for key_val_gt in key_val_gt_list:
                #         iou_2d = get_iou2d(bbox2D_pd, key_val_gt[3])
                #         iou_2d_list.append(iou_2d)
                #     max_iou = max(iou_2d_list)
                #     if max_iou == 0:
                #         continue
                #     print(key)
                #     max_index = iou_2d_list.index(max_iou)
                #     bbox2D_gt = key_val_gt_list[max_index][3]
                #     draw_2d_box(img, bbox2D_pd[0], bbox2D_pd[1], bbox2D_pd[2], bbox2D_pd[3], c=(0, 255, 0))
                #     draw_2d_box(img, bbox2D_gt[0], bbox2D_gt[1], bbox2D_gt[2], bbox2D_gt[3] , c=(0, 0, 255))
                        #     for key_label_list in key_label_lists:
                        #         key_bbox2d = key_label_list[3]
                        #         iou_2d = get_iou2d(bbox2D.tolist(), key_bbox2d)
                        #         iou_2d_list.append(iou_2d)
                        #     # print("iou_2d_list" + str(len(iou_2d_list)))
                    # cv2.imshow("img", img)
                    # cv2.waitKey()

                    # bbox2D[:2] = affine_transform(bbox2D[:2], trans_box)  # (112, 112)에 맞게 BR 점의 위치를 옮겨줌
                    # bbox2D[2:] = affine_transform(bbox2D[2:], trans_box)




        # for txt in pd_list:
        #     tmp = txt[1:-3].split(',')
        #     cat_2d = tmp[0][1:-1]
        #     score_2d = float(tmp[1])
        #     bbox2D = np.array([float(tmp[2][2:]), float(tmp[3]), float(tmp[4]), float(tmp[5])], dtype = np.float32)
        #
        #     trans_box = get_affine_transform(c, s, 0, [img_w, img_h])
        #     bbox2D[:2] = affine_transform(bbox2D[:2], trans_box)  # (112, 112)에 맞게 BR 점의 위치를 옮겨줌
        #     bbox2D[2:] = affine_transform(bbox2D[2:], trans_box)
        #     # #np.clip함수를 사용하여 bbox행렬의 범위를 output_w, output-h 내의 범위로 바꿔줌
        #
        #     bbox2D[[0, 2]] = np.clip(bbox2D[[0, 2]], 0, img_w - 1)
        #     bbox2D[[1, 3]] = np.clip(bbox2D[[1, 3]], 0, img_h - 1)
        #     #draw_2d_pd
        #     draw_2d_box(img, bbox2D[0], bbox2D[1], bbox2D[2], bbox2D[3], c=(0, 255, 0))
        #
        #     # cv2.imshow("img", img)
        #     # cv2.waitKey()
        #
        #     # 하나의 pd_2d에 대해서 그것과 동일한 cat에 대해서 all gt_2d
        #     key_label_lists = gt_dict_per_img[cat_2d]
        #     # for key in gt_dict_per_img :
        #     #     if cat_2d == key:
        #     #         key_label_lists =gt_dict_per_img[key]
        #     # 그 all gt_2d와 하나의 pd_2d iou를 모두 비교하여 iou_2d_list에 저장함.
        #     iou_2d_list = []
        #     # pd_2d가 찾은 cat의 gt_dict가 비어있는 상황.
        #     if len(key_label_lists) == 0:
        #         continue
        #
        #     for key_label_list in key_label_lists:
        #         key_bbox2d = key_label_list[3]
        #         iou_2d = get_iou2d(bbox2D.tolist(), key_bbox2d)
        #         iou_2d_list.append(iou_2d)
        #     # print("iou_2d_list" + str(len(iou_2d_list)))
        #
        #     assert len(key_label_lists) == len(iou_2d_list)
        #     # print("iou_2d_list")
        #     # print(iou_2d_list)
        #     max_iou = max(iou_2d_list)
        #     #동일한 cat임은 인정. 그런데 겹침이 하나도 없는 곳.
        #     if int(max_iou) == 0:
        #         continue
        #     key_index = iou_2d_list.index(max_iou)
        #     bbox2D_gt = key_label_lists[key_index][3]
        #     draw_2d_box(img, bbox2D_gt[0], bbox2D_gt[1], bbox2D_gt[2], bbox2D_gt[3], c= (0, 0, 255))
        #
        #     pd3_label = gt_dict_per_img[cat_2d][key_index]
        #     print(pd3_label)
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
