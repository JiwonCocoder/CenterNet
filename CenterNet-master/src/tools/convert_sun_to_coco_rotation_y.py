from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import io
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib
import pickle

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
from utils.ddd_utils import compute_box_3d, project_to_image, project_to_image_sun2, project_to_image_sun, alpha2rot_y, compute_box_3d_sun, compute_box_3d_sun_2, compute_box_3d_sun_3,compute_box_3d_sun_4,compute_box_3d_sun_5,compute_box_3d_sun_6
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d, draw_box_3d_sun
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
        #if i == 2:  # P2에 대해서 \n제거하고, 맨 앞 item(P2)이름빼고, 빼고 나머지 값들을 배열로
        #calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
        line = line[1:-1].split(',')
        calib = np.array(line, dtype=np.float32)
        calib = calib.reshape(3, 4)  # 그냥 앞에서부터 순서대로 잘라서 넣는다.
        return calib
def read_Rtilt(Rtilt_path):
    f = open(Rtilt_path, 'r')
    for i, line in enumerate(f):
        #if i == 2:  # P2에 대해서 \n제거하고, 맨 앞 item(P2)이름빼고, 빼고 나머지 값들을 배열로
        #calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
        line = line[1:-1].split(',')
        Rtilt = np.array(line, dtype=np.float32)
        Rtilt = Rtilt.reshape(3, 3)  # 그냥 앞에서부터 순서대로 잘라서 넣는다.
        return Rtilt

'''
cats = ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck', 'Person_sitting',
        'Tram', 'Misc', 'DontCare']
'''

cats = ['counter', 'paper_cutter', 'papers', 'lamination_machine', 'box', 'drawer',
        'garbage_bin', 'scanner', 'poster', 'chair', 'tv', 'bed', 'night_stand', 'ottoman', 'dresser_mirror','dresser', 'lamp', 'sofa', 'sofa_bed']
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


cat_ids = {cat: i + 1 for i, cat in
           enumerate(class_final)}  # cat_ids는 dict형. keyword로 cat_name : value로 번호 1~8까지가 object. 9가 don't care
# cat_info = [{"name": "pedestrian", "id": 1}, {"name": "vehicle", "id": 2}]
F = 721
H = 384  # 375
W = 1248  # 1242
EXT = [45.75, -0.34, 0.005]
CALIB = np.array([[F, 0, W / 2, EXT[0]], [0, F, H / 2, EXT[1]],
                  [0, 0, 1, EXT[2]]], dtype=np.float32)


cat_info = []  # list형. 원소들은 dict형 - name: cat명, id: 그 cat명의 index
for i, cat in enumerate(cats):
    cat_info.append({'name': cat, 'id': i + 1})
# network에 맞는 dir를 찾아가도록 추가해줌.
for SPLIT in SPLITS:
    #image_set_path = '../../data/kitti/' + 'ImageSets_{}/'.format(SPLIT)
    ann_dir = DATA_PATH + 'labelIndexing/'
    #calib_dir = DATA_PATH + '{}/calib/'  # 오류인듯: 뒤에 .format(SPLIT)이 붙어야 할듯
    calib_dir = DATA_PATH + 'calibIndexing/'  # 수정
    Rtilt_dir = DATA_PATH +'RtiltIndexing/'
    basis_dir = DATA_PATH + 'basisIndexing/'
    img_dir = DATA_PATH + 'imgIndexing/'
    splits = ['train', 'val']
    # splits = ['trainval', 'test']
    calib_type = {'train': 'training', 'val': 'training', 'trainval': 'training',
                  'test': 'testing'}

    for split in splits:
        ret = {'images': [], 'annotations': [], "categories": cat_info}
        #image_set = open(image_set_path + 'pair_Index_to_sunPath.txt', 'r')
        image_set = list(range(10335))
        image_to_id = {}
        for line in image_set:  # train.txt에 있는 line을 str로 하나씩 읽어들임
            # if line[-1] == '\n':  # line(str)의 마지막에 \n있으면 빼주고.
            #     line = line[1:-2].split(',')
            # image_id = line[0] # ex.000003 → 3
            # image_path = line[1][1:-1]

            # image_name = image_path.split('/')[-1]
            #calib_path = calib_dir.format(calib_type[split]) + '{}.txt'.format(image_path)
            image_id = line
            img_path = img_dir + '{}.jpg'.format(image_id)
            calib_path = calib_dir + '{}.txt'.format(image_id)
            Rtilt_path = Rtilt_dir + '{}.txt'.format(image_id)
            basis_path = basis_dir + '{}.txt'.format(image_id)
            calib = read_clib(calib_path)
            Rtilt = read_Rtilt(Rtilt_path)
            basis = read_Rtilt(basis_path)
            image_info = {'file_name': img_path,
                          'id': image_id,
                          'calib': calib.tolist()}  # 1자로 펴서 넣어준다.
            ret['images'].append(image_info)
            if split == 'test':
                continue
            ann_path = ann_dir + '{}.txt'.format(image_id)  # line = train.txt = image의 번호
            # if split == 'val':
            #   os.system('cp {} {}/'.format(ann_path, VAL_PATH))
            anns = open(ann_path, 'r')

            if DEBUG:
                image = cv2.imread(image_info['file_name'])

            for ann_ind, txt in enumerate(anns):  # 한줄씩 처리한다는 의미: ann_ind는 object 갯수에 대한 index.
                tmp = txt[1:-2].split(',')  # white space제거
                #cat_index = cat_ids[tmp[0][1:-1]]  # 가장 앞에 있는 str은 object의 이름
                cat_id = tmp[0][1:-1]
                if cat_id in classes:
                    if cat_id == "books":
                        cat_id = "book"
                        pass
                    elif cat_id == "bottle" or cat_id == "bowl" or cat_id == "plate" or cat_id == "cup" or cat_id == "mug":
                        cat_id = "dishware"
                        pass
                    elif cat_id == "cabinet" or cat_id == "kitchen_cabinet" or cat_id == "file_cabinet" or cat_id == "night_stand":
                        cat_id = "drawer"
                        pass
                    elif cat_id == "kitchen_counter":
                        cat_id = "counter"
                        pass
                    elif cat_id == "picture":
                        label = "painting"
                        pass
                    elif cat_id == "plants" or cat_id == "flowers":
                        cat_id = "plant"
                        pass
                    elif cat_id == "bookshelf":
                        cat_id = "shelf"
                        pass
                    elif cat_id == "sofa_chair":
                        cat_id = "sofa"
                        pass
                    elif cat_id == "endtable" or cat_id == "coffee_table" or cat_id == "dining_table":
                        cat_id = "table"
                        pass
                    pass
                else:
                    continue
                    pass
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
                #tmp순서: centroid_x, centroid_y, centroid_z, length, height, width

                #방법1) x,z,y수넛로 값을 집어 넣기
                location = [float(tmp[1]), float(tmp[2]), float(tmp[3])] ##z는 이미 - 되어져서 온 상황.

                # 3D object location x,z,y in camera coords. [m]
                #왜냐하면 SUN-RGBD는 x,z평면을 이용하기 때문에, location의 (x,y)위치에 SUN-RGBD의 (x,z)가 들어가줘야한다.
                #dim = [float(tmp[5]), float(tmp[6]), float(tmp[4])]  # 3D object dimensions: height, width, length [m]
                dim = [float(tmp[4]), float(tmp[5]), float(tmp[6])]  # 3D object dimensions: x길이, y길이, z길이
                print("dim"+str(dim))
                rotation_y = float(tmp[7])  # Rotation around Y-axis in camera coords. [-Pi; Pi]
                ann = {'image_id': image_id,
                       'id': int(len(ret['annotations']) + 1),
                 #      'category_id': cat_index,
                       'category_name' : cat_id,
                       'dim': dim,
                      # 'bbox': _bbox_to_coco_bbox(bbox),
                       'depth': location[2],
                      # 'alpha': alpha,
                      # 'truncated': truncated,
                      # 'occluded': occluded,
                       'location': location,
                       'rotation_y': rotation_y}
                ret['annotations'].append(ann)
                if DEBUG and cat_id in class_final:
                    #EXT = [45.75, -0.34, 0.005]
                    # calib[0][3] = 45.75
                    # calib[1][3] = -0.34
                    print(cat_id)
                    # calib[2][3] = 0.005
                    box_3d = compute_box_3d_sun_6(dim, location, rotation_y, Rtilt, basis)
                    box_2d = project_to_image(box_3d, calib)  # 이제 3d bounding box를 image에 투영시킴
                    # print('box_2d', box_2d)

                    image = draw_box_3d_sun(image, box_2d)
                    # x = (bbox[0] + bbox[2]) / 2
                    '''
                    print('rot_y, alpha2rot_y, dlt', tmp[0], 
                          rotation_y, alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0]),
                          np.cos(
                            rotation_y - alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0])))
                    '''
                    depth = np.array([location[2]], dtype=np.float32)
                    #pt_2d = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                    #                 dtype=np.float32)
                    #pt_3d = unproject_2d_to_3d(pt_2d, depth, calib)
                    #pt_3d[1] += dim[0] / 2  # because the position of  KITTI is defined as the center of the bottom face
                    #print('pt_3d', pt_3d)
                    #print('location', location)
            if DEBUG:
                cv2.imshow('image', image)
                cv2.waitKey()

        print("# images: ", len(ret['images']))
        print("# annotations: ", len(ret['annotations']))
        # import pdb; pdb.set_trace()
        out_path = '{}/annotations/kitti_{}_{}.json'.format(DATA_PATH, SPLIT, split)
        json.dump(ret, open(out_path, 'w'))

