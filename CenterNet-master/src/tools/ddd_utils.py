from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.spatial.transform import Rotation as Rot

import numpy as np
import cv2
import math
def compute_box_3d(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  l, w, h = dim[2], dim[1], dim[0]
  x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  y_corners = [0,0,0,0,-h,-h,-h,-h]
  z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

  corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
  corners_3d = np.dot(R, corners)  #R_rot *x_ref_coord(camera_coordinate에서의 좌표값들)
  temp = np.array(location, dtype=np.float32).reshape(3, 1)
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1) #camera좌표 (0,0)에서 시작했었으니까, 물체 center point로 평행이동시킴
  return corners_3d.transpose(1, 0)


#축변환 후, rotation, Rtilted_centroid한 상황.
#Rtilt는 적용되지 않은 상황.
def compute_box_3d_sun_10(dim, location, rotation_y, Rtilt):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  x, y, z = dim[0], dim[1], dim[2]
  x_corners = [-x, x, x, -x, -x, x, x, -x]
  y_corners = [y, y, -y, -y, y, y, -y, -y]
  z_corners = [z, z, z, z, -z, -z, -z, -z]
  corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
  corners = np.dot(np.transpose(Rtilt), corners)
  corners = np.transpose(corners)  # (8,3)
  sun_permutation = [0, 2, 1]  # (x,z,y)
  index = np.argsort(sun_permutation)
  corners = corners[:, index]  # (8,3) #(x,z,y)
  corners[:, 1] *= -1  # (x,z,-y)

  #R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  print("before_np.argsort" + str(R))
  print("R[:,0]:" + str(R[:, 0]))
  print("-abs:" + str(-abs(R[:, 0])))
  print("argsort:" + str(np.argsort(-abs(R[:, 0]))))
  # rot_inds = np.argsort(-abs(R[:, 0]))

  # when R matrix가 y중심일 때
  rot_inds = order_rotation_y_matrix(R)
  rot_mat = R[rot_inds, :]

  #when R matrix가 z중심일 때
  # rot_inds = np.argsort(-abs(R[0, :]))
  # R =R[:, rot_inds]

  #Rtilt for y
#   Rtilt_index = [0, 2, 1]
#   Rtilt_y = np.copy(Rtilt)
#   Rtilt_y= Rtilt_y[:, Rtilt_index]
#   Rtilt_y = Rtilt_y[Rtilt_index, :]
#   Rtilt_y = np.transpose(Rtilt_y)
#   Rtilt_y[1, :] *= -1
# #
  corners_3d = np.dot(rot_mat, corners.transpose(1,0))  #R_rot *x_ref_coord(camera_coordinate에서의 좌표값들)
  #corners_3d = np.dot(Rtilt_y, corners.transpose(1, 0))  # R_rot *x_ref_coord(camera_coordinate에서의 좌표값들)
  location = np.dot(np.transpose(Rtilt), location)
  location_changed = [location[i] for i in sun_permutation]
  location_changed[1] *= -1
  temp = np.array(location_changed, dtype=np.float32).reshape(3, 1)
  corners_3d = corners_3d + np.array(location_changed, dtype=np.float32).reshape(3, 1) #camera좌표 (0,0)에서 시작했었으니까, 물체 center point로 평행이동시킴

  return corners_3d.transpose(1, 0)
#rotation (y/x)로 바꾼 후, sun_test_img를 위해 compute_box_3d_sun_8에서 Rtilt를 뺀 상황
def compute_box_3d_sun_9(dim, location, theta_c):
  # print("roatation_y:" +str(rotation_y))
  c, s = np.cos(theta_c), np.sin(theta_c)
  print("rotation_y:" + str(theta_c) + "각도는" + str(theta_c * 180 / math.pi))

  #R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  print("before_np.argsort" + str(R))
  print("R[:,0]:" + str(R[:, 0]))
  print("-abs:" + str(-abs(R[:, 0])))
  print("argsort:" + str(np.argsort(-abs(R[:, 0]))))
  # rot_inds = np.argsort(-abs(R[:, 0]))

  # when R matrix가 y중심일 때
  rot_inds = order_rotation_y_matrix(R)
  rot_mat = R[rot_inds, :]

  #when R matrix가 z중심일 때
  # rot_inds = np.argsort(-abs(R[0, :]))
  # R =R[:, rot_inds]

  print("after_np.argsort" + str(R))
  rot_coeffs = abs(np.array(dim))
  x = rot_coeffs[0]
  y = rot_coeffs[1]
  z = rot_coeffs[2]
  x_corners = [-x, x, x, -x, -x, x, x, -x]
  y_corners = [y, y, y, y, -y, -y, -y, -y]
  z_corners = [z, z, -z, -z, z, z, -z, -z]
  # y_corners = [y, y, -y, -y, y, y, -y, -y]
  # z_corners = [z, z, z, z, -z, -z, -z, -z]
  corners2 = np.array([x_corners, y_corners, z_corners], dtype=np.float32)

  corners_3d = np.dot(rot_mat, corners2)  # (3,8)
  # #Rotation 행렬을 그대로 사용하는 상황
  # corners_3d = np.dot(R, corners2)  #(3,8)

  temp = np.array(location, dtype=np.float32).reshape(3, 1)

  # y ↔ z 축변환_1 (생략)
  # corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3,1) #(3,8)
  # corners_3d = corners_3d.transpose(1, 0)

  #Rtilt (생략)
  # corners_3d = np.dot(np.transpose(Rtilt), np.transpose(corners_3d[:, 0:3]))
  # corners_3d = corners_3d.transpose(1, 0)

  # y ↔ z 축변환_2 (생략)
  # sun_permutation = [0,2,1]
  # index = np.argsort(sun_permutation)
  #
  # corners_3d = corners_3d[:, index]
  # corners_3d[:, 1] *= -1
  return corners_3d.transpose(1, 0)

# rotation을 y/x로 바꾼 후, 그거에 맞는 compute_box function
def compute_box_3d_sun_8(dim, location, theta_c, Rtilt):
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
  corners_3d = np.dot(np.transpose(Rtilt), np.transpose(corners_3d[:, 0:3]))
  corners_3d = corners_3d.transpose(1, 0)
  sun_permutation = [0,2,1]
  index = np.argsort(sun_permutation)

  corners_3d = corners_3d[:, index]
  corners_3d[:, 1] *= -1
  return corners_3d

def compute_box_3d_sun_7(dim, location, rotation_y):  # x,y,z를 순서대로
  #print("roatation_y:" +str(rotation_y))
  c, s = np.cos(-rotation_y), np.sin(-rotation_y)
  print("rotation_y:" + str(-rotation_y) + "각도는" + str(-rotation_y * 180 / math.pi))

  #R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

  print("R[:,0]:"+ str(R[:,0]))
  print("-abs:" + str(-abs(R[:, 0])))
  print("argsort:" + str(np.argsort(-abs(R[:, 0]))))
  #rot_inds = np.argsort(-abs(R[:, 0]))
  rot_inds = order_rotation_y_matrix(R)
 # rot_inds = [0, 1, 2]
  rot_mat = R[rot_inds, :]
  print("after_np.argsort" + str(rot_mat))
  rot_coeffs = abs(np.array(dim))
  x = rot_coeffs[0]
  y = rot_coeffs[1]
  z = rot_coeffs[2]
  x_corners = [-x, x, x, -x, -x, x, x, -x]
  y_corners = [y, y, -y, -y, y, y, -y, -y]
  z_corners = [z, z, z, z, -z, -z, -z, -z]
  corners2 = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
##z용으로바꾸기!!!
  corners_3d = np.dot(R, corners2)  #R_rot *x_ref_coord(camera_coordinate에서의 좌표값들)
  temp = np.array(location, dtype=np.float32).reshape(3, 1)
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1) #camera좌표 (0,0)에서 시작했었으니까, 물체 center point로 평행이동시킴
  return corners_3d.transpose(1, 0)

def compute_box_3d_sun(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  #l, w, h = dim[2], dim[1], dim[0]
  l, w, h = dim[0], dim[1], dim[2]
  #w는 세로를 담당
  x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  y_corners = [0,0,0,0,-h,-h,-h,-h]
  z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
  corners = np.array([x_corners, z_corners, y_corners], dtype=np.float32)
  corners_3d = np.dot(R, corners)  #R_rot *x_ref_coord(camera_coordinate에서의 좌표값들)
  temp = np.array(location, dtype=np.float32).reshape(3, 1)
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1) #camera좌표 (0,0)에서 시작했었으니까, 물체 center point로 평행이동시킴
  return corners_3d.transpose(1, 0)


def compute_box_3d_sun_2(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3

  #현재 dim이랑 location이 x,z,y. length, width, height순으로 설정되어 있는 상태
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  #l, w, h = dim[2], dim[1], dim[0]
  l, w, h = dim[0], dim[1], dim[2]
  #w는 세로를 담당
  x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  #y_corners = [0,0,0,0,-h,-h,-h,-h]
  #y_corners = [h, h, h, h, 0, 0, 0, 0]
  y_corners = [-h, -h, -h, -h, 0, 0, 0, 0]
  z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
  corners = np.array([x_corners, z_corners, y_corners], dtype=np.float32)
  corners_3d = np.dot(R, corners)  #R_rot *x_ref_coord(camera_coordinate에서의 좌표값들)
  temp = np.array(location, dtype=np.float32).reshape(3, 1)
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1) #camera좌표 (0,0)에서 시작했었으니까, 물체 center point로 평행이동시킴
  return corners_3d.transpose(1, 0)

def compute_box_3d_sun_3(dim, location, rotation_y, Rtilt_ori):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  #R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  #R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  #R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
  R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
  # R_tilt = np.zeros((3,3))
  # R_tilt[0] = [0.979589, 0.012593, -0.200614]
  # R_tilt[2] = [0.012593, 0.992231, 0.123772]
  # R_tilt[1] = [0.200614, -0.123772, 0.97182]
  #Rtilt_ori = np.array([[0.979589, 0.012593, -0.200614],[0.012593, 0.992231, 0.123772],[0.200614, -0.123772, 0.97182]], dtype=np.float32)

  R_tilt = Rtilt_ori
  #R_tilt = np.matrix(Rtilt_ori).I

  #l, w, h = dim[2], dim[1], dim[0]
  l, h, w = dim[0]*2 , dim[1]*2, dim[2]*2
  x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  #y_corners = [0, 0, 0, 0, h, h, h, h]
  #y_corners = [0,0,0,0,-h,-h,-h,-h]
  #y_corners = [-h, -h, -h, -h, 0, 0, 0, 0 ]
  y_corners = [h, h, h, h, 0, 0, 0, 0]
  z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
  #z_corners = [w, 0, 0, w, w, 0, 0, w]
  ##추가 부분
  corners_new = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
  corners_new = R_tilt @ corners_new
  corners_new = corners_new.transpose(1, 0)
  sun_permutation = [0, 2, 1]
  i = np.argsort(sun_permutation)
  corners = corners_new[:, i]
  corners = corners.transpose(1,0)
  ##
  #corners = np.array([x_corners, z_corners, y_corners], dtype=np.float32)
  #location_sun = location[z], location[x], location[y]
  location_sun = [location[0], location[2], location[1]]
  #corners = np.dot(R_tilt, corners)
  corners_3d = np.dot(R, corners)  #R_rot *x_ref_coord(camera_coordinate에서의 좌표값들)
  temp = np.array(location_sun, dtype=np.float32).reshape(3, 1)
  corners_3d = corners_3d + np.array(location_sun, dtype=np.float32).reshape(3, 1) #camera좌표 (0,0)에서 시작했었으니까, 물체 center point로 평행이동시킴
  corners_3d.transpose(1,0)[:, 1] = -corners_3d.transpose(1,0)[:, 1]
  #corners_3d = np.dot(R_tilt, corners_3d)

  return corners_3d.transpose(1, 0)

def compute_box_3d_sun_4(dim, location, rotation_y, Rtilt_ori): #x,y,z를 순서대로
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  #R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  #R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  #R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
  rot_inds = np.argsort(-abs(R[:, 0]))
  R = R[rot_inds,:]
  # R_tilt = np.zeros((3,3))
  # R_tilt[0] = [0.979589, 0.012593, -0.200614]
  # R_tilt[2] = [0.012593, 0.992231, 0.123772]
  # R_tilt[1] = [0.200614, -0.123772, 0.97182]
  #Rtilt_ori = np.array([[0.979589, 0.012593, -0.200614],[0.012593, 0.992231, 0.123772],[0.200614, -0.123772, 0.97182]], dtype=np.float32)

  #R_tilt = np.matrix(Rtilt_ori).I
  #l, w, h = dim[2], dim[1], dim[0]
  l, h, w = dim[0]*2 , dim[1]*2 , dim[2]*2
  # x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  # y_corners = [0,0,0,0,-h,-h,-h,-h]
  # z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
  x_corners = [-l/2, l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2]
  y_corners = [h/2, h/2, -h/2, -h/2, h/2,h/2,-h/2,-h/2]
  z_corners = [w/2, w/2, w/2, w/2, -w/2, -w/2, -w/2, -w/2]
  corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
  corners = np.dot(corners.transpose(1, 0), R)  #R_rot *x_ref_coord(camera_coordinate에서의 좌표값들)
  corners = corners.transpose(1, 0)
  #location_sun = location[z], location[x], location[y]
  location_sun = [location[0], location[1], location[2]]
  corners = corners + np.array(location_sun, dtype=np.float32).reshape(3, 1) #camera좌표 (0,0)에서 시작했었으니까, 물체 center point로 평행이동시킴
  corners = np.dot(Rtilt_ori, corners)
  corners_3d = corners.transpose(1, 0)
  sun_permutation = [0, 2, 1]
  i = np.argsort(sun_permutation)
  corners_3d = corners_3d[:, i]
  temp = np.array(location_sun, dtype=np.float32).reshape(3, 1)
  corners_3d[:, 1] = -corners_3d[:, 1]
  #corners_3d = np.dot(R_tilt, corners_3d)
  return corners_3d #(8,3인상황)

#실제로 작동되는 상황
def compute_box_3d_sun_5(dim, location, rotation_y, Rtilt, basis):  # x,y,z를 순서대로
  corners = np.zeros((8, 3))
  corners[0, :] = -basis[0, :] * dim[0] + basis[1, :] * dim[1] + basis[2, :] * dim[2]
  corners[1, :] = basis[0, :] * dim[0] + basis[1, :] * dim[1] + basis[2, :] * dim[2]
  corners[2, :] = basis[0, :] * dim[0] - basis[1, :] * dim[1] + basis[2, :] * dim[2]
  corners[3, :] = -basis[0, :] * dim[0] - basis[1, :] * dim[1] + basis[2, :] * dim[2]
  corners[4, :] = -basis[0, :] * dim[0] + basis[1, :] * dim[1] - basis[2, :] * dim[2]
  corners[5, :] = basis[0, :] * dim[0] + basis[1, :] * dim[1] - basis[2, :] * dim[2]
  corners[6, :] = basis[0, :] * dim[0] - basis[1, :] * dim[1] - basis[2, :] * dim[2]
  corners[7, :] = -basis[0, :] * dim[0] - basis[1, :] * dim[1] - basis[2, :] * dim[2]


  corners += np.matlib.repmat(location, 8, 1)


  corners = Rtilt @ corners.transpose(1, 0)
  corners = corners.transpose(1, 0)
  corners = corners.astype('float32')


  sun_permutation = [0, 2, 1]  # (x,z,y)
  index = np.argsort(sun_permutation)
  corners = corners[:, index]
  return corners
def order_rotation_y_matrix(R):
  if(-abs(R[0,0]) > -abs(R[2, 0])) :
    rot_inds = [2, 1, 0]
  else:
    rot_inds = [0, 1, 2]
  return rot_inds
#이게 예전 데이터셋에 사용했던 최종본
def compute_box_3d_sun_6(dim, location, rotation_y, Rtilt):  # x,y,z를 순서대로
  corners2 = np.zeros((8,3))
  #print("roatation_y:" +str(rotation_y))
  c, s = np.cos(-rotation_y), np.sin(-rotation_y) #왜냐하면 theta_c가 들어왔으니까
  print("rotation_y:" + str(rotation_y) + "각도는" + str(rotation_y * 180 / math.pi))

  #R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

  print("R[:,0]:"+ str(R[:,0]))
  print("-abs:" + str(-abs(R[:, 0])))
  print("argsort:" + str(np.argsort(-abs(R[:, 0]))))
  #rot_inds = np.argsort(-abs(R[:, 0]))
  rot_inds = order_rotation_y_matrix(R)
 # rot_inds = [0, 1, 2]
  rot_mat = R[rot_inds, :]
  print("after_np.argsort" + str(rot_mat))
  rot_coeffs = abs(np.array(dim))
  corners2[0, 0] = -rot_coeffs[0]
  corners2[0, 1] = rot_coeffs[1]
  corners2[0, 2] = rot_coeffs[2]

  corners2[1, 0] = rot_coeffs[0]
  corners2[1, 1] = rot_coeffs[1]
  corners2[1, 2] = rot_coeffs[2]

  corners2[2, 0] = rot_coeffs[0]
  corners2[2, 1] = -rot_coeffs[1]
  corners2[2, 2] = rot_coeffs[2]

  corners2[3, 0] = -rot_coeffs[0]
  corners2[3, 1] = -rot_coeffs[1]
  corners2[3, 2] = rot_coeffs[2]

  corners2[4, 0] = -rot_coeffs[0]
  corners2[4, 1] = rot_coeffs[1]
  corners2[4, 2] = -rot_coeffs[2]

  corners2[5, 0] = rot_coeffs[0]
  corners2[5, 1] = rot_coeffs[1]
  corners2[5, 2] = -rot_coeffs[2]

  corners2[6, 0] = rot_coeffs[0]
  corners2[6, 1] = -rot_coeffs[1]
  corners2[6, 2] = -rot_coeffs[2]

  corners2[7, 0] = -rot_coeffs[0]
  corners2[7, 1] = -rot_coeffs[1]
  corners2[7, 2] = -rot_coeffs[2]
  #corners2 = rot_mat @ corners2.transpose(1, 0) #(3,8)
  #corners2 = corners2.transpose(1, 0) #(8,3)
  #location[2] = -location[2]


  #print("corners2" + str(corners2))

  #location[1] = -location[1]

  corners2 = corners2.astype('float32')
  sun_permutation = [0, 2, 1]  # (x,z,y)
  index = np.argsort(sun_permutation)
  #축변환
  corners2 = corners2[:, index]
  print("Rtilt:"+str(Rtilt))
  Rtilt_index = [0, 2, 1]
  Rtilt= Rtilt[:, Rtilt_index]
  Rtilt = Rtilt[Rtilt_index, :]
  #Rtilt[:, 2] = -1 * Rtilt[:, 2]
  print("Rtilt_change:" + str(Rtilt))
  #rotation
  corners2 = corners2 @ rot_mat #(8,3)
  location_changed = [location[i] for i in sun_permutation]

  corners2 += np.matlib.repmat(location_changed, 8, 1) #(8,3)

  corners2 = Rtilt @ corners2.transpose(1, 0)  # (3,8)
  corners2 = corners2.transpose(1,0)   # (8,3)

  #corners2 = corners2[:, index] #이렇게 하면 답이 나오긴 함.
  print("corners2" + str(corners2))

  print("changed X and Y:" + str(location_changed))

  #corners2 = corners2.transpose(1, 0) #(8,3)
  #corners2[:, 1] = - corners2[:, 1]

  return corners2 #(8,3)
def project_to_image(pts_3d, P):
  # pts_3d: n x 3
  # P: 3 x 4
  # return: n x 2
  pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1) #homo_coord를 만드는 과정: (x,y,z,1)로
  pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
  #normalized image plnae에서의 카메라좌표상의 위치를 project시킨 상황.
  pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]

  pts_2d = np.array(pts_2d, dtype=np.int64)
  # import pdb; pdb.set_trace()
  return pts_2d

def project_to_centerPoint(centerPoint, P):
  # pts_3d: n x 3
  # P: 3 x 4
  # return: n x 2
  pts_3d_homo = np.concatenate(
    [centerPoint, np.ones((centerPoint.shape[0], 1), dtype=np.float32)], axis=1) #homo_coord를 만드는 과정: (x,y,z,1)로
  pts_2d = np.dot(P[0:2], pts_3d_homo.transpose(1, 0)).transpose(1, 0)
  #normalized image plnae에서의 카메라좌표상의 위치를 project시킨 상황.
  pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]

  # import pdb; pdb.set_trace()
  return pts_2d

def project_to_image_sun(pts_3d, P):
  pts_3d[:, 1] = -pts_3d[:, 1]
  # pts_3d: n x 3
  # P: 3 x 4
  # return: n x 2
  P_test = np.array(
    [[7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02, 4.575831000000e+01],
     [0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02, -3.454157000000e-01],
     [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 4.981016000000e-03]],
    dtype=np.float32)
  pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1) #homo_coord를 만드는 과정: (x,y,z,1)로
  pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
  pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
  # import pdb; pdb.set_trace()
  return pts_2d

def project_to_image_sun2(pts_3d, P):
  pts_3d[:, 1] = -pts_3d[:, 1]
  # pts_3d: n x 3
  # P: 3 x 4
  # return: n x 2
  #P[0][3] = 1000
  pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1) #homo_coord를 만드는 과정: (x,y,z,1)로
  pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
  pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
  # import pdb; pdb.set_trace()
  return pts_2d
def compute_orientation_3d(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 2 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  orientation_3d = np.array([[0, dim[2]], [0, 0], [0, 0]], dtype=np.float32)
  orientation_3d = np.dot(R, orientation_3d)
  orientation_3d = orientation_3d + \
                   np.array(location, dtype=np.float32).reshape(3, 1)
  return orientation_3d.transpose(1, 0)

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
      cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
               (corners[f[(j+1)%4], 0], corners[f[(j+1)%4], 1]), c, 1   , lineType=cv2.LINE_AA)
      #show
      # cv2.imshow("img", image)
      # cv2.waitKey()
    # if ind_f == 0: #암면에 대해서는 대각선으로 표시
    #   cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
    #            (corners[f[2], 0], corners[f[2], 1]), c, 1, lineType=cv2.LINE_AA)
    #   cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
    #            (corners[f[3], 0], corners[f[3], 1]), c, 1, lineType=cv2.LINE_AA)
  return image

def draw_box_3d_sun(image, corners, image_id, c=(255, 255, 255)):
  # face_idx = [[7,6,2,3], #앞
  #             [7,4,0,3], #왼
  #             [4,5,1,0],  #뒤
  #             [1,5,6,2]] #오
  face_idx = [[1,2,6,5],
              [2, 3, 7, 6],
              [0,1,5,4], #앞
              [3,0,4,7]] #오
  for ind_f in range(3, -1, -1):
    f = face_idx[ind_f]
    for j in range(4):
      #print(corners.shape)
      #print(corners)
      cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
               (corners[f[(j+1)%4], 0], corners[f[(j+1)%4], 1]), c, 2, lineType=cv2.LINE_AA)
      # if ind_f == 0:  # 암면에 대해서는 대각선으로 표시
      #   cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
      #            (corners[f[2], 0], corners[f[2], 1]), c, 1, lineType=cv2.LINE_AA)
      #   cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
      #            (corners[f[3], 0], corners[f[3], 1]), c, 1, lineType=cv2.LINE_AA)
      # cv2.imshow(str(image_id), image)
      # cv2.waitKey()
  return image

def draw_box_3d_world(image, corners, image_id, c=(0, 0, 255)):
  # face_idx = [[7,6,2,3], #앞
  #             [7,4,0,3], #왼
  #             [4,5,1,0],  #뒤
  #             [1,5,6,2]] #오
  face_idx = [[1,2,6,5],
              [2, 3, 7, 6],
              [0,1,5,4], #앞
              [3,0,4,7]] #오
  for ind_f in range(3, -1, -1):
    f = face_idx[ind_f]
    for j in range(4):
      #print(corners.shape)
      #print(corners)
      cv2.line(image, (int(corners[f[j], 0]), int(corners[f[j], 1])),
               (int(corners[f[(j+1)%4], 0]), int(corners[f[(j+1)%4], 1])), c, 2, lineType=cv2.LINE_AA)
      if ind_f == 0:  # 암면에 대해서는 대각선으로 표시
        cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1])),
                 (int(corners[f[2], 0]), int(corners[f[2], 1])), c, 1, lineType=cv2.LINE_AA)
        cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1])),
                 (int(corners[f[3], 0]), int(corners[f[3], 1])), c, 1, lineType=cv2.LINE_AA)
      # cv2.imshow(str(image_id), image)
      # cv2.waitKey()
  return image
def unproject_2d_to_3d(pt_2d, depth, P):
  # pts_2d: 2
  # depth: 1
  # P: 3 x 4
  # return: 3
  z = depth - P[2, 3]
  x = (pt_2d[0] * depth - P[0, 3] - P[0, 2] * z) / P[0, 0]
  y = (pt_2d[1] * depth - P[1, 3] - P[1, 2] * z) / P[1, 1]
  pt_3d = np.array([x, y, z], dtype=np.float32)
  print("unproject_2d_to_3d")
  print(depth)
  print(pt_2d)
  print(x,y,z)
  print(P)
  return pt_3d

def alpha2rot_y(alpha, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + np.arctan2(x - cx, fx)
    if rot_y > np.pi:
      rot_y -= 2 * np.pi
    if rot_y < -np.pi:
      rot_y += 2 * np.pi
    return rot_y

def rot_y2alpha(rot_y, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    alpha = rot_y - np.arctan2(x - cx, fx)
    if alpha > np.pi:
      alpha -= 2 * np.pi
    if alpha < -np.pi:
      alpha += 2 * np.pi
    return alpha


def ddd2locrot(center, alpha, dim, depth, calib):
  # single image
  locations = unproject_2d_to_3d(center, depth, calib)
  #locations[1] += dim[0] / 2
  rotation_y = alpha2rot_y(alpha, center[0], calib[0, 2], calib[0, 0])
  return locations, rotation_y

def project_3d_bbox(location, dim, rotation_y, calib):
  box_3d = compute_box_3d(dim, location, rotation_y)
  box_2d = project_to_image(box_3d, calib)
  return box_2d


if __name__ == '__main__':
  calib = np.array(
    [[7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02, 4.575831000000e+01],
     [0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02, -3.454157000000e-01],
     [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 4.981016000000e-03]],
    dtype=np.float32)
  alpha = -0.20
  tl = np.array([712.40, 143.00], dtype=np.float32)
  br = np.array([810.73, 307.92], dtype=np.float32)
  ct = (tl + br) / 2
  rotation_y = 0.01
  print('alpha2rot_y', alpha2rot_y(alpha, ct[0], calib[0, 2], calib[0, 0]))
  print('rotation_y', rotation_y)
  print('rot_y2alppha', rot_y2alpha(rotation_y, ct[0], calib[0,2], calib[0,0]))