import math
import json
import numpy as np
import cv2
from array import array
from numpy.linalg import inv
import os
from IoUPython import get_3d_box, box3d_iou


def project_to_image(pts_3d, P):
  # pts_3d: n x 3
  # P: 3 x 4
  # return: n x 2
  pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1) #homo_coord를 만드는 과정: (x,y,z,1)로
  pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
  #normalized image plnae에서의 카메라좌표상의 위치를 project시킨 상황.
  pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
  return pts_2d
def draw_box_3d(image, corners, c=(128, 128, 128)):
  #kitti용
  # face_idx = [[0,1,5,4], #앞
  #             [1,2,6,5], #왼
  #             [2,3,7,6],  #뒤
  #             [3,0,4,7]] #오
  #sun rgbd용
  face_idx = [[1,2,6,5], #앞
              [6,5,4,7], #왼
              [7,4,0,3],  #뒤
              [3,0,1,2]] #오
  # face_idx = [[[2,3,7,6]],
  #              [3,0,4,7],
  #              [[0,1,5,4]],
  #              [1,2,6,5]]
  for ind_f in range(3, -1, -1):
    f = face_idx[ind_f]
    for j in range(4):
      cv2.line(image, (int(corners[f[j], 0]), int(corners[f[j], 1])),
               (int(corners[f[(j+1)%4], 0]), int(corners[f[(j+1)%4], 1])), c, 2, lineType=cv2.LINE_AA)
      #show
      # cv2.imshow("img", image)
      # cv2.waitKey()
    # if ind_f == 0: #암면에 대해서는 대각선으로 표시
    #   cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
    #            (corners[f[2], 0], corners[f[2], 1]), c, 1, lineType=cv2.LINE_AA)
    #   cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
    #            (corners[f[3], 0], corners[f[3], 1]), c, 1, lineType=cv2.LINE_AA)
  return image

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
        Rtilt = np.array(line, dtype=np.float32)
        Rtilt = Rtilt.reshape(3, 3)  # 그냥 앞에서부터 순서대로 잘라서 넣는다.
        return Rtilt

def calc_stats(val):
    cat_max = np.max(val)
    cat_min = np.min(val)
    cat_mean = np.mean(val)
    cat_std = np.std(val)
    return cat_max, cat_min, cat_mean, cat_std


def get_loc_unRtilt_for_cam(location, Rtilt_cam):
    inversed_Rtilt_cam = inv(Rtilt_cam)
    location_unRtilt_for_cam = np.dot(inversed_Rtilt_cam, location)
    return location_unRtilt_for_cam

def get_Rtilt_cam(Rtilt):
    Rtilt = np.transpose(Rtilt)

    Rtilt_cam = np.zeros((3, 3))
    Rtilt_cam[0, 0] = Rtilt[0, 0]
    Rtilt_cam[0, 1] = -Rtilt[0, 2]
    Rtilt_cam[0, 2] = Rtilt[0, 1]

    Rtilt_cam[1, 0] = -Rtilt[2, 0]
    Rtilt_cam[1, 1] = Rtilt[2, 2]
    Rtilt_cam[1, 2] = -Rtilt[2, 1]

    Rtilt_cam[2, 0] = Rtilt[1, 0]
    Rtilt_cam[2, 1] = -Rtilt[1, 2]
    Rtilt_cam[2, 2] = Rtilt[1, 1]
    return Rtilt_cam

DATA_PATH = '../../data/SUNRGBD/'

ann_dir_dt = DATA_PATH + 'labelIndexing_3d_3_test/'
ann_dir_gt = DATA_PATH + 'labelIndexing_ct3d/'

Rtilt_dir = DATA_PATH + 'RtiltIndexing/'
calib_dir = DATA_PATH + 'calibIndexing/'  # 수정
img_dir = DATA_PATH + 'imgIndexing/'
#여기에 test에 사용할 img의 index들이 들어있음
splits = ['test_3d_2']

ID = {"bed": 0, "drawer": 1, "chair": 2,
      "counter": 3, "door": 4, "painting": 5, "pillow": 6,
      "shelf": 7, "sofa": 8, "toilet": 9, "tv": 10}

cat_info = []
# i = (0+1) ~ (10+1)

for i, cat in enumerate(ID):
    cat_info.append({'name': cat, 'id': i + 1})
print(cat_info)

cat_dim_dict = {"bed": [], "drawer": [], "chair": [],
                "counter": [], "door": [], "painting": [], "pillow": [],
                "shelf": [], "sofa": [], "toilet": [], "tv": []}
cat_dep_dict = {"bed": [], "drawer": [], "chair": [],
                "counter": [], "door": [], "painting": [], "pillow": [],
                "shelf": [], "sofa": [], "toilet": [], "tv": []}
#먼저 detect한 label들을 dict에 저장해 놓기
skip_img_id_list = []
for split in splits:
    #이게 원래 json만들 때 쓰려고 했던 dict형
    # ret = {'images':[], 'loc':[], 'dim':[], 'rot_y':[]}
    # ret = {'images': [], 'annotations': [], "categories": cat_info}
    ret = {'cat': [], 'loc': [], 'dim': [], 'rot_y': []}
    ret_gt = {'cat': [], 'loc': [], 'dim': [], 'rot_y': []}
    img_index_from_txt = open(DATA_PATH + '{}.txt'.format(split), 'r')
    for line in img_index_from_txt:
        image_id = line[:-1]
        print("image_id" + str(image_id))
        img_path = img_dir + '{}.jpg'.format(image_id)
        calib_path = calib_dir + '{}.txt'.format(image_id)
        Rtilt_path = Rtilt_dir + '{}.txt'.format(image_id)
        calib = read_clib(calib_path)
        Rtilt = read_Rtilt(Rtilt_path)

        # image_info = {'file_name': img_path.split('/')[-1],
        #         #               'id': int(image_id),
        #         #               'calib': calib.tolist()}  # 1자로 펴서 넣어준다.
        #         # ret['images'].append(image_info)
        # ann_path_dt = ann_dir_dt + '{}.txt'.format('test')
        ann_path_dt = ann_dir_dt + '{}.txt'.format(image_id)  # line = train.txt = image의 번호
        ann_path_gt = ann_dir_gt + '{}.txt'.format(image_id)
        anns_dt = open(ann_path_dt, 'r')
        anns_dt2 = open(ann_path_dt, 'r')
        anns_gt = open(ann_path_gt, 'r')

        image = cv2.imread(img_path)
        if os.path.isfile(ann_path_dt):
            print("here")
            #readline을 했을 때 empty인 것들은 나중에 skip_img로 만들어서
            dt = anns_dt.read().splitlines()
            print(dt)
            if len(dt) == 0:
                skip_img_id_list.append(int(image_id))
            else:
                cat_name_gt_list = []
                loc_gt_list = []
                dim_gt_list = []
                rot_y_gt_list = []
                sun_permutation = [0,2,1]
                for ann_ind, txt in enumerate(anns_gt):
                    tmp = txt[1:-2].split(',')
                    cat_name_gt = tmp[0][1:-1]
                    location_gt = [float(tmp[1]), float(tmp[2]), float(tmp[3])]
                    dim_gt =[float(tmp[4]), float(tmp[5]), float(tmp[6])]
                    rotation_y_gt = float(tmp[7])
                    sun_permutation = [0, 2, 1]
                    location_tilt = np.dot(np.transpose(Rtilt), location_gt)
                    location_changed = [location_tilt[i] for i in sun_permutation]
                    location_changed[1] *= -1
                    dim_changed = [dim_gt[i] * 2 for i in sun_permutation]  # 왜냐하면 dimension은 크기니까
                    cat_name_gt_list.append(cat_name_gt)
                    loc_gt_list.append(location_changed)
                    dim_gt_list.append(dim_changed)
                    rot_y_gt_list.append(rotation_y_gt)
                if len(cat_name_gt_list) == len(dim_gt_list):
                    gt_num = len(cat_name_gt_list)
                for ann_ind, txt in enumerate(anns_dt2):
                    tmp = txt[1:-2].split(',')
                    print(tmp)
                    cat_name = tmp[0][1:-1]
                    location = [float(tmp[1]), float(tmp[2]), float(tmp[3])]
                    dim = [float(tmp[4]), float(tmp[5]), float(tmp[6])]
                    rotation_y = float(tmp[7])

                    ret['cat'] = cat_name
                    ret['loc'] = location
                    ret['dim'] = dim
                    ret['rot_y'] = rotation_y
                    Rtilt_cam = get_Rtilt_cam(Rtilt)
                    loc_unRtilt = get_loc_unRtilt_for_cam(location, Rtilt_cam)
                    corners_3d_predict = get_3d_box(dim, rotation_y, loc_unRtilt) #(8,3)
                    corners_3d_predict_Rtilt = np.dot(Rtilt_cam, corners_3d_predict.transpose(1,0))
                    corners_3d_predict_Rtilt = corners_3d_predict_Rtilt.transpose(1,0)# (8,3)

                    for i in range(gt_num):
                        if(cat_name == cat_name_gt_list[i]):
                            corners_3d_ground = get_3d_box(dim_gt_list[i], rot_y_gt_list[i], loc_gt_list[i]) #(8,3)
                            corners_3d_ground_Rtilt = np.dot(Rtilt_cam, corners_3d_ground.transpose(1, 0))  #(3,8)
                            # corners_3d_ground_Rtilt = corners_3d_ground_Rtilt.transpose(1, 0) #(8,3)
                            # (IOU_3d, IOU_2d) = box3d_iou(corners_3d_predict_Rtilt, corners_3d_ground_Rtilt)
                            # print(IOU_3d, IOU_2d)  # 3d IoU/ 2d IoU of BEV(bird eye's view)
                            # print(corners_3d_predict_Rtilt)
                            box_2d_gt = project_to_image(corners_3d_ground_Rtilt.transpose(1,0), calib)
                            image = draw_box_3d(image, box_2d_gt, c=(128, 128, 128))
                            xmin_projected = int(box_2d_gt.transpose()[0].min())
                            xmax_projected = int(box_2d_gt.transpose()[0].max())
                            ymin_projected = int(box_2d_gt.transpose()[1].min())
                            ymax_projected = int(box_2d_gt.transpose()[1].max())
                            cv2.line(image, (xmin_projected, ymin_projected), (xmax_projected, ymin_projected), (0, 255, 255), 2, lineType=cv2.LINE_AA)
                            cv2.line(image, (xmax_projected, ymin_projected), (xmax_projected, ymax_projected), (0, 255, 255), 2, lineType=cv2.LINE_AA)
                            cv2.line(image, (xmin_projected, ymax_projected), (xmin_projected, ymin_projected), (0, 255, 255), 2, lineType=cv2.LINE_AA)
                            cv2.line(image, (xmax_projected, ymax_projected), (xmin_projected, ymax_projected), (0, 255, 255), 2, lineType=cv2.LINE_AA)
                            cv2.imshow(str(image_id), image)
                            cv2.waitKey()

    skip_img_id_list.sort()
    print(len(skip_img_id_list))
    print(skip_img_id_list)
            #만약에 아무것도 없으면, if문에 걸림

            # for ann_ind, txt in enumerate(anns_dt):  # 한줄씩 처리한다는 의미: ann_ind는 object 갯수에 대한 index.
            #     print("here")