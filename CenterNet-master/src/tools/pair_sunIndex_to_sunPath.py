from scipy import io
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy import matlib
import numpy as np
import cv2
import math
import sys
import os
# np.set_printoptions(threshold=sys.maxsize)

def flip_towards_viewer(normals, points):
    mat = np.matlib.repmat(np.sqrt(np.sum(points*points, 1)), 3, 1)
    points = points / mat
    # print(points)
    proj = np.sum(points * normals, 1)
    flip = proj > 0
    normals[flip, :] = -normals[flip, :]
    return normals


def make_kitti_calib (mat_file):
    for range_index in range(4) :
        data = mat_file.popitem()
        if(data[0] == 'SUNRGBDMeta'):
            lists_index_img_path =[]
            destination_path = '../../data/SUNRGBD/'
            imageset_path = destination_path+ 'imgIndexing/'
            for i in range(10335):
                #     for j in range(13):
                #
                #           """       data[1][0][image_index][index]
                #             0. sequenceName     (string)
                #             1. groundtruth3DBB  (9 element struct)
                #             2. Rtilt            (3x3 double)
                #             3. K                (3x3 double)
                #             4. depthpath        (string)
                #             5. rgbpath          (string)
                #             6. anno_extrinsics  (3x3 double)
                #             7. depthname        (string)
                #             8. rgbname          (string)
                #             9. sensorType       (string : kv1 / kv2 / realsense / xtion)
                #             10. valid           (전부 1.)
                #             11. gtCorner3D      (3xN double, 없는것도 있음)
                #             12. groundtruth2DBB (4 element struct)
                #         """
                #list = []
                path = data[1][0][i][5][0]
                color_raw = cv2.imread(path, -1)
                cv2.imwrite('{}/{}.jpg'.format(imageset_path, i), color_raw)


        """
            data[1][0][index] --> index 번째 data. ( 0 <= index <= 10335 )
            data[1][0][index][index2] --> index 번째 data의 index2 번째 data. Matlab에서 column에 해당. 13개.
        """
        #calib 파일을 이미지 이름으로 만들어주는 상황.



    #     #color_raw = (530, 730, 3)
    #     color_raw = cv2.imread(data[1][0][image_index][5][0], cv2.COLOR_RGB2BGR) # mat형태로 imag 값을 받아옴
    #     #depth_raw = (530, 730)
    #
    #     depth_raw = cv2.imread(data[1][0][image_index][4][0], -1) #depth 값을 받아옴 (1개)
    #
    #     #
    #     # """
    #     #     uint8 color_raw data to float32 range(0,1)
    #     # """
    #     #
    #     '''
    #     필요없을 듯
    #     rgb = np.reshape(color_raw, (len(color_raw)*len(color_raw[0]), 3))
    #     rgb = rgb.astype("float32")
    #     rgb = rgb / 255
    #     '''
    #     #
    #     #
    #     # """
    #     #     Make 3d point cloud by using depth_raw
    #     # """
    #     #depth를 m단위로 변환해줌
    #     # depthInpaint = (530, 730)
    #     depthInpaint = (depth_raw>>3) | (depth_raw<<(16-3))
    #     depthInpaint = depthInpaint.astype("float32")
    #     depthInpaint = depthInpaint / 1000
    #     # 8m이상이면 무시
    #     # for row in depthInpaint :
    #     #     for ele in row :
    #     #         ele = 8 if ele > 8 else ele
    #     #         pass
    #     #     pass
    #     #이건 카메라 파라미터로,
    #     K = data[1][0][image_index][3]
    #     cx = K[0][2]
    #     cy = K[1][2]
    #     fx = K[0][0]
    #     fy = K[1][1]
    #
    #     range_x = np.arange(1, len(depth_raw[0])+1)
    #     range_y = np.arange(1, len(depth_raw)+1)
    #
    #     x, y = np.meshgrid(range_x, range_y)
    #
    #     x3 = (x-cx)*depthInpaint*1/fx
    #     y3 = (y-cy)*depthInpaint*1/fy
    #     z3 = depthInpaint
    #
    #     x3 = np.reshape(x3, len(x3)*len(x3[0]))
    #     y3 = np.reshape(y3, len(y3)*len(y3[0]))
    #     z3 = np.reshape(z3, len(z3)*len(z3[0]))
    #     #pointsMat = np.vstack((x3,-y3,z3))
    #     pointsMat = np.vstack((x3,z3,-y3))
    #     #point cloud 그려주기 위해서 (시작)
    #     # # remove nan
    #     nan_index = []
    #     for i in range(len(x3)):
    #         # if x3[i] != 0 or y3[i] != 0 or z3[i] != 0:
    #         if x3[i] == 0 and y3[i] == 0 and z3[i] == 0:
    #             nan_index.append(i)
    #             pass
    #         pass
    #     pointsMat = np.delete(pointsMat, nan_index, axis=1)
    #     rgb = np.delete(rgb, nan_index, axis=0)
    #
    #     Rtilt = data[1][0][image_index][2]
    #     point3d = Rtilt @ pointsMat
    #
    #     """
    #         Random sampling.
    #         260631 --> 10000.
    #     """
    #     sample_size = np.random.randint(len(point3d[0]), size=10000)
    #     x3 = point3d[0, sample_size]
    #     y3 = point3d[2, sample_size]
    #     z3 = point3d[1, sample_size]
    #     rgb = rgb[sample_size, :]
    #


if __name__ == '__main__':
    fig = pyplot.figure()
    ax = Axes3D(fig)

    mat_file = io.loadmat('C:/SUNRGBDMeta/SUNRGBDMeta.mat')
    make_kitti_calib(mat_file)
