from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import cv2
DATA_PATH = '../../data/kitti/'
DEBUG = False
# VAL_PATH = DATA_PATH + 'training/label_val/'
import os
SPLITS = ['3dop', 'subcnn'] 
import _init_paths
from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d
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

def _bbox_to_coco_bbox(bbox):
  return [(bbox[0]), (bbox[1]),
          (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]

def read_clib(calib_path):
  f = open(calib_path, 'r')
  for i, line in enumerate(f):
    if i == 2: #P2에 대해서 \n제거하고, 맨 앞꺼(P2)빼고 나머지 값들을 배열로
      calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
      calib = calib.reshape(3, 4) #그냥 앞에서부터 순서대로 잘라서 넣는다.
      return calib

cats = ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person_sitting',
        'Tram', 'Misc', 'DontCare']
cat_ids = {cat: i + 1 for i, cat in enumerate(cats)} # cat_ids는 dict형. keyword로 cat_name : value로 번호 1~8까지가 object. 9가 don't care
# cat_info = [{"name": "pedestrian", "id": 1}, {"name": "vehicle", "id": 2}]
F = 721
H = 384 # 375
W = 1248 # 1242
EXT = [45.75, -0.34, 0.005]
CALIB = np.array([[F, 0, W / 2, EXT[0]], [0, F, H / 2, EXT[1]], 
                  [0, 0, 1, EXT[2]]], dtype=np.float32)

cat_info = [] #list형. 원소들은 dict형 - name: cat명, id: 그 cat명의 index
for i, cat in enumerate(cats):
  cat_info.append({'name': cat, 'id': i + 1})
#network에 맞는 dir를 찾아가도록 추가해줌.
for SPLIT in SPLITS:
  image_set_path = DATA_PATH + 'ImageSets_{}/'.format(SPLIT)
  ann_dir = DATA_PATH + 'training/label_2/'
  calib_dir = DATA_PATH + '{}/calib/' #오류인듯: 뒤에 .format(SPLIT)이 붙어야 할듯
  splits = ['train', 'val']
  # splits = ['trainval', 'test']
  calib_type = {'train': 'training', 'val': 'training', 'trainval': 'training',
                'test': 'testing'}

  for split in splits:
    ret = {'images': [], 'annotations': [], "categories": cat_info}
    image_set = open(image_set_path + '{}.txt'.format(split), 'r')
    image_to_id = {}
    for line in image_set: #train.txt에 있는 line을 str로 하나씩 읽어들임
      if line[-1] == '\n': #line(str)의 마지막에 \n있으면 빼주고.
        line = line[:-1]
      image_id = int(line)
      calib_path = calib_dir.format(calib_type[split]) + '{}.txt'.format(line)
      calib = read_clib(calib_path)
      image_info = {'file_name': '{}.png'.format(line),
                    'id': int(image_id),
                    'calib': calib.tolist()} #1자로 펴서 넣어준다.
      ret['images'].append(image_info)
      if split == 'test':
        continue
      ann_path = ann_dir + '{}.txt'.format(line) #line = train.txt = image의 번호
      # if split == 'val':
      #   os.system('cp {} {}/'.format(ann_path, VAL_PATH))
      anns = open(ann_path, 'r')
      
      if DEBUG:
        image = cv2.imread(
          DATA_PATH + 'images/trainval/' + image_info['file_name'])

      for ann_ind, txt in enumerate(anns): #한줄씩 처리한다는 의미: ann_ind는 object 갯수에 대한 index.
        tmp = txt[:-1].split(' ') #white space제거
        cat_id = cat_ids[tmp[0]] #가장 앞에 있는 str은 object의 이름
        truncated = int(float(tmp[1])) #truncated refers to the object leaving image boundaries. 화면에 남아있는 정도를 의미
        occluded = int(tmp[2]) #가려진 정도를 의미 0 = fully visible, 1 = partly occluded 2 = largely occluded, 3 = unknown
        alpha = float(tmp[3]) #Observation angle of object, ranging [-Pi; Pi]
        bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])] # (0-based) bounding box of the object: Left, top, right, bottom image coordinates
        dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])] #3D object dimensions: height, width, length [m]
        location = [float(tmp[11]), float(tmp[12]), float(tmp[13])] #3D object location x,y,z in camera coords. [m]
        rotation_y = float(tmp[14]) # Rotation around Y-axis in camera coords. [-Pi; Pi]

        ann = {'image_id': image_id,
               'id': int(len(ret['annotations']) + 1),
               'category_id': cat_id,
               'dim': dim,
               'bbox': _bbox_to_coco_bbox(bbox),
               'depth': location[2],
               'alpha': alpha,
               'truncated': truncated,
               'occluded': occluded,
               'location': location,
               'rotation_y': rotation_y}
        ret['annotations'].append(ann)
        if DEBUG and tmp[0] != 'DontCare':
          box_3d = compute_box_3d(dim, location, rotation_y)
          box_2d = project_to_image(box_3d, calib)
          # print('box_2d', box_2d)
          image = draw_box_3d(image, box_2d)
          x = (bbox[0] + bbox[2]) / 2
          '''
          print('rot_y, alpha2rot_y, dlt', tmp[0], 
                rotation_y, alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0]),
                np.cos(
                  rotation_y - alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0])))
          '''
          depth = np.array([location[2]], dtype=np.float32)
          pt_2d = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                            dtype=np.float32)
          pt_3d = unproject_2d_to_3d(pt_2d, depth, calib)
          pt_3d[1] += dim[0] / 2 # because the position of  KITTI is defined as the center of the bottom face
          print('pt_3d', pt_3d)
          print('location', location)
      if DEBUG:
        cv2.imshow('image', image)
        cv2.waitKey()


    print("# images: ", len(ret['images']))
    print("# annotations: ", len(ret['annotations']))
    # import pdb; pdb.set_trace()
    out_path = '{}/annotations/kitti_{}_{}.json'.format(DATA_PATH, SPLIT, split)
    json.dump(ret, open(out_path, 'w'))
  
