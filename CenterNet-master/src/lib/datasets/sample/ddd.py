from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import pycocotools.coco as coco
import numpy as np
import torch
import json
import cv2
import os
import math
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
import pycocotools.coco as coco

class DddDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _convert_alpha(self, alpha):
    return math.radians(alpha + 45) if self.alpha_in_degree else alpha

  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    img = cv2.imread(img_path)
    #cv2.imshow("original", img)
    #cv2.waitKey()
    if 'calib' in img_info:
      calib = np.array(img_info['calib'], dtype=np.float32)
    else:
      calib = self.calib
    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.]) #width, height순으로 이미지의 center point를 구함.
    if self.opt.keep_res:
      s = np.array([self.opt.input_w, self.opt.input_h], dtype=np.int32)
    else:
      s = np.array([width, height], dtype=np.int32) #그때끄때마다 다른상황.
    
    aug = False
    if self.split == 'train' and np.random.random() < self.opt.aug_ddd:
      #augmentation되는 상황.
      aug = True
      sf = self.opt.scale
      cf = self.opt.shift
      # img_size바꿔주고
      s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      #center 이동시키고
      c[0] += img.shape[1] * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
      c[1] += img.shape[0] * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
    trans_input = get_affine_transform( # scale만큼 크기를 바꿔주고, rot만큼 회전시키고,
      c, s, 0, [self.opt.input_w, self.opt.input_h])
    inp = cv2.warpAffine(img, trans_input, 
                         (self.opt.input_w, self.opt.input_h),
                         flags=cv2.INTER_LINEAR)
    #아마 cv2.warpAffine을 해야 cv2 img form으로 바뀌는 듯
    #cv2.imshow("augmentation", inp)
    #cv2.waitKey()
    inp = (inp.astype(np.float32) / 255.)
    # if self.split == 'train' and not self.opt.no_color_aug:
    #   color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1) #WHC to CWH

    num_classes = self.opt.num_classes
    #trans_output도 동일하게 affine_transform을 시켜줌. 들어온 input size를 output size로 바꾸어주는 어파일 행렬을 만듬
    trans_output = get_affine_transform(
      c, s, 0, [self.opt.output_w, self.opt.output_h])
   #opt.output크기를 내가 보고 싶어서
    out = cv2.warpAffine(img, trans_output,
                         (self.opt.output_w, self.opt.output_h),
                         flags=cv2.INTER_LINEAR)
    #cv2.imshow("output", out)
    #cv2.waitKey()
    hm = np.zeros(
      (num_classes, self.opt.output_h, self.opt.output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    dep = np.zeros((self.max_objs, 1), dtype=np.float32)
    rotbin = np.zeros((self.max_objs, 2), dtype=np.int64)
    rotres = np.zeros((self.max_objs, 2), dtype=np.float32)
    dim = np.zeros((self.max_objs, 3), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    rot_mask = np.zeros((self.max_objs), dtype=np.uint8)

    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian
    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      #bbox: 이제가 우리가 아는 (xmin, ymin, xmax, ymax)로 이루어진 행렬
      bbox = self._coco_box_to_bbox(ann['bbox'])
      cls_id = int(self.cat_ids[ann['category_id']])
      if cls_id <= -99:
        continue
      # if flipped:
      #   bbox[[0, 2]] = width - bbox[[2, 0]] - 1
      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      #np.clip함수를 사용하여 bbox행렬의 범위를 output_w, output-h 내의 범위로 바꿔줌
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if h > 0 and w > 0:
        radius = gaussian_radius((h, w))
        radius = max(0, int(radius))
        #box의 center point를 구하는 공식이 맞음.
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        if cls_id < 0:
          ignore_id = [_ for _ in range(num_classes)] \
                      if cls_id == - 1 else  [- cls_id - 2]
          if self.opt.rect_mask:
            hm[ignore_id, int(bbox[1]): int(bbox[3]) + 1, 
              int(bbox[0]): int(bbox[2]) + 1] = 0.9999
          else:
            for cc in ignore_id:
              draw_gaussian(hm[cc], ct, radius)
           # hm[ignore_id, ct_int[1], ct_int[0]] = 0.9999
          continue
       #heatmap의 center에다가 가우시안 필터를 씌워줌
        draw_gaussian(hm[cls_id], ct, radius)
        a = np.where(hm[0]==1)
        b = np.where(hm[1] ==1)
        c = np.where(hm[2] == 1)
        wh[k] = 1. * w, 1. * h #h랑 w는 box size이다. 따라서 max_object(제한)된 곳에 쌓아둠.
        #gt_det라는 list에다가 3d obj detector에 필요한 일자들을 넣어줌.
        #center [x, y ,1], orientation, depth, diemsion(w,h,1을 배열로 만들고, 그걸 list로 만들어서 이 모든 걸 하나의 원소로 넣어줌
        #append에다가 [] + [] 로 계속 연결시키면, 하나의 원소로 들어가는 듯.
        gt_det.append([ct[0], ct[1], 1] + \
                      self._alpha_to_8(self._convert_alpha(ann['alpha'])) + \
                      [ann['depth']] + (np.array(ann['dim']) / 1).tolist() + [cls_id])
        if self.opt.reg_bbox:
          #가장 마지막 gt_det에 대해서 cls_id만을 제외한 원소값들을 사용하는데, 거기다가 [w,h] 를 붙인다음에 마지막에 cls_id가 오도록.
          gt_det[-1] = gt_det[-1][:-1] + [w, h] + [gt_det[-1][-1]]
        # if (not self.opt.car_only) or cls_id == 1: # Only estimate ADD for cars !!!
        #indicator function : 아마 물체가 있따면
        if 1:
          alpha = self._convert_alpha(ann['alpha'])
          # print('img_id cls_id alpha rot_y', img_path, cls_id, alpha, ann['rotation_y'])
          if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
            rotbin[k, 0] = 1 #bin의 0 index를 1로
            rotres[k, 0] = alpha - (-0.5 * np.pi)
          if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
            rotbin[k, 1] = 1 #bin의 1 index를 1로
            rotres[k, 1] = alpha - (0.5 * np.pi)
          dep[k] = ann['depth']
          dim[k] = ann['dim']
          # print('        cat dim', cls_id, dim[k])
          ind[k] = ct_int[1] * self.opt.output_w + ct_int[0]
          reg[k] = ct - ct_int
          reg_mask[k] = 1 if not aug else 0
          rot_mask[k] = 1
    # print('gt_det', gt_det)
    # print('')
    ret = {'input': inp, 'hm': hm, 'dep': dep, 'dim': dim, 'ind': ind, 
           'rotbin': rotbin, 'rotres': rotres, 'reg_mask': reg_mask,
           'rot_mask': rot_mask}
    if self.opt.reg_bbox:
      ret.update({'wh': wh})
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not ('train' in self.split):
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 18), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'calib': calib,
              'image_path': img_path, 'img_id': img_id}
      ret['meta'] = meta
    
    return ret

  def _alpha_to_8(self, alpha):
    # return [alpha, 0, 0, 0, 0, 0, 0, 0]
    ret = [0, 0, 0, 1, 0, 0, 0, 1]
    if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.: #앞쪽 bin이 담당하는 영역
      r = alpha - (-0.5 * np.pi)
      ret[1] = 1
      ret[2], ret[3] = np.sin(r), np.cos(r)
    if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.: #뒷쪽 bin이 담당하는 영역
      r = alpha - (0.5 * np.pi)
      ret[5] = 1
      ret[6], ret[7] = np.sin(r), np.cos(r)
    return ret
