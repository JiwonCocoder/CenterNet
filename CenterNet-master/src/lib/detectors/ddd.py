from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import os
from external.nms import soft_nms
from models.decode import ddd_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ddd_post_process
from utils.debugger import Debugger
from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d

from .base_detector import BaseDetector

class DddDetector(BaseDetector):
  def __init__(self, opt):
    super(DddDetector, self).__init__(opt)
    # self.calib = np.array([[707.0493, 0, 604.0814, 45.75831],
    #                        [0, 707.0493, 180.5066, -0.3454157],
    #                        [0, 0, 1., 0.004981016]], dtype=np.float32)
    # test_img calib 수동 입력상황
    #self.calib = os.path.join(opt.data_dir, 'SUNRGBD/calibIndexing')
    self.calib = np.array([[529.5, 0, 365, 0],
                           [0, 529.5, 265, 0],
                           [0, 0, 1., 0]], dtype=np.float32)


  def pre_process(self, image, scale, calib=None):
    height, width = image.shape[0:2]
    #change_point
    #sun
    #height, width = int(448), int(448)
    #original(1280, 384) → network input(1280, 384)
    #original(1280, 384) → network input(1280, 384)
    inp_height, inp_width = self.opt.input_h, self.opt.input_w
    c = np.array([width / 2, height / 2], dtype=np.float32)
    if self.opt.keep_res: #test시에는 여기로 들어가서, opt.input에서 지정한 값을 사용함
      s = np.array([inp_width, inp_height], dtype=np.int32)
    else: #지금 해상도가 일치하지 않으므로, s에 original크기를 넣어주고
      s = np.array([width, height], dtype=np.int32)
    #
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height]) #(2,3)
    resized_image = image #cv2.resize(image, (width, height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height), #인자를 전달할 때, src, M, dst
      flags=cv2.INTER_LINEAR)
    #cv2.imshow("test_input", inp_image)
    #show
    #cv2.waitKey()
    inp_image = (inp_image.astype(np.float32) / 255.) # 픽셀값을 정규화시킴
    inp_image = (inp_image - self.mean) / self.std #표준정규분포를 따르도록
    images = inp_image.transpose(2, 0, 1)[np.newaxis, ...] #WHC 순으로
    calib = np.array(calib, dtype=np.float32) if calib is not None \
            else self.calib
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 
            'out_height': inp_height // self.opt.down_ratio, 
            'out_width': inp_width // self.opt.down_ratio,
            'calib': calib}
    return images, meta
  
  def process(self, images, return_time=False):
    with torch.no_grad():
      torch.cuda.synchronize()
      output = self.model(images)[-1]
      #output['hm'] = heaetmap
      output['hm'] = output['hm'].sigmoid_()
      output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
      wh = output['wh'] if self.opt.reg_bbox else None
      reg = output['reg'] if self.opt.reg_offset else None
      torch.cuda.synchronize()
      forward_time = time.time()
      #tilt 추가
      # dets = ddd_decode(output['hm'], output['rot'], output['dep'],
      #                     output['dim'], output['tilt'],wh=wh, reg=reg, K=self.opt.K)
      dets = ddd_decode(output['hm'], output['rot'], output['dep'],
                          output['dim'], wh=wh, reg=reg, K=self.opt.K)

    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    detections = ddd_post_process(
      dets.copy(), [meta['c']], [meta['s']], [meta['calib']], self.opt)
    self.this_calib = meta['calib']
    return detections[0]

  def merge_outputs(self, detections):
    results = detections[0]
    for j in range(1, self.num_classes + 1):
      if len(results[j] > 0):
        keep_inds = (results[j][:, -1] > self.opt.peak_thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, images, dets, output, image_id, scale=1):
    dets = dets.detach().cpu().numpy()
    img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = ((img * self.std + self.mean) * 255).astype(np.uint8)
    pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
    pred2 = debugger.gen_colormap_dep(output['dep'][0].detach().cpu().numpy())
    pred3 = debugger.gen_colormap_dim(output['dim'][0].detach().cpu().numpy())
    print(output['hm'][0])
    #pred2 = debugger.gen_colormap(output['hm'])
    debugger.add_blend_img(img, pred, 'pred_hm')
    #debugger.add_blend_img(img, pred2, 'pred_depth')
    #debugger.add_blend_img(img, pred3, 'pred_dim')

    debugger.add_ct_detection(
      img, dets[0], image_id, show_box=self.opt.reg_bbox,
      center_thresh=self.opt.vis_thresh, img_id='det_pred')
  
  def show_results(self, debugger, image, results, image_id):
    debugger.add_3d_detection(
      image, results, self.this_calib, image_id,
      center_thresh=self.opt.vis_thresh, img_id='add_pred')
    print("results")
    #debugger.add_bird_view(
     # results, center_thresh=self.opt.vis_thresh, img_id='bird_pred')
#show 없애고 save
    debugger.show_all_imgs(pause=self.pause)
#     cv2.imwrite(
#   'C:\\obj_detection\\CenterNet-master\\CenterNet-master\\data\\SUNRGBD\\results_pred\\' + str(image_id) + '.jpg',
#   image)
