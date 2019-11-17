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
            i = 0
            # (1) rgbpath를 보기 위해서
            path = data[1][0][i][5][0]
            print(path)
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
            K = np.array(data[1][0][i][3])
            new_K = np.zeros((K.shape[0],K.shape[1]+1))
            new_K[:,:-1] = K #(3,3)은 원래데이터
            R = np.array(data[1][0][i][2])
            new_R = np.zeros((R.shape[0]+1,R.shape[1]+1))
            new_R[:-1, :-1] = R
            new_R[3][3] = 1

            #list.extend(new_K.reshape(-1))
            list = new_K.reshape(-1).tolist()
            img_path = data[1][0][i][5]
            #f = open('../data/SUNRGBD/calib/'+str(img_path).split('/')[-1].split('.')[0]+'.txt', 'w')

            #del list[:] #리스트 내 원소들만 지워주기
            list.clear()
            depth_raw = cv2.imread(data[1][0][i][4][0], -1) #depth 값을 받아옴 (1개)
            depthInpaint = (depth_raw>>3) | (depth_raw<<(16-3))
            depthInpaint = depthInpaint.astype("float32")
            depthInpaint = depthInpaint / 1000
            color_raw = cv2.imread(data[1][0][i][5][0], cv2.COLOR_RGB2BGR)  # mat형태로 imag 값을 받아옴
            # cv2.imshow("hihi", color_raw)
            # cv2.waitKey()
            rgb = np.reshape(color_raw, (len(color_raw) * len(color_raw[0]), 3))
            rgb = rgb.astype("float32")
            rgb = rgb / 255
            K = data[1][0][i][3]
            cx = K[0][2]
            cy = K[1][2]
            fx = K[0][0]
            fy = K[1][1]

            range_x = np.arange(1, len(depth_raw[0])+1)
            range_y = np.arange(1, len(depth_raw)+1)

            x, y = np.meshgrid(range_x, range_y)

            x3 = (x-cx)*depthInpaint*1/fx
            y3 = (y-cy)*depthInpaint*1/fy
            z3 = depthInpaint

            x3 = np.reshape(x3, len(x3)*len(x3[0]))
            y3 = np.reshape(y3, len(y3)*len(y3[0]))
            z3 = np.reshape(z3, len(z3)*len(z3[0]))
            #pointsMat = np.vstack((x3,-y3,z3))
            pointsMat = np.vstack((x3,z3,-y3))
            # # remove nan
            nan_index = []
            for j in range(len(x3)):
                # if x3[i] != 0 or y3[i] != 0 or z3[i] != 0:
                if x3[j] == 0 and y3[j] == 0 and z3[j] == 0:
                    nan_index.append(j)
                    pass
                pass
            pointsMat = np.delete(pointsMat, nan_index, axis=1)
            rgb = np.delete(rgb, nan_index, axis=0)
            Rtilt = data[1][0][i][2]
            #Rtilt = np.matrix(Rtilt).I
            point3d = Rtilt @ pointsMat
            """
                Random sampling.
                260631 --> 10000.
            """
            sample_size = np.random.randint(len(point3d[0]), size=10000)
            x3 = point3d[0, sample_size]
            y3 = point3d[2, sample_size]
            z3 = point3d[1, sample_size]
            rgb = rgb[sample_size, :]
            """
                Visualize
            """
            label_lists = []
            for groundtruth3DBB in data[1][0][i][1]:
                #한 이미지에 들어있는 물체의 갯수만큼 반복문이 실행됨
                for items in groundtruth3DBB:
                    #label_list = []

                    """
                        items = data[1][0][image_index][1][(groundtruth3DBB)]
                        items[index]
                        0. basis            (3x3 double)
                        1. coeffs           (1x3 double)
                        2. centroid         (1x3 double)
                        3. classname        (string) #label_name
                        4. labelname        (?)
                        5. sequenceName     (string)
                        6. orientation      (1x3 double)
                        7. gtBb2D           (1x4 double)
                        8. label            ()
                    """
                    print(items)
                    K = data[1][0][i][3]
                    cx = K[0][2]
                    cy = K[1][2]
                    fx = K[0][0]
                    fy = K[1][1]

                    range_x = np.arange(1, len(depth_raw[0]) + 1)
                    range_y = np.arange(1, len(depth_raw) + 1)

                    x, y = np.meshgrid(range_x, range_y)

                    x3 = (x - cx) * depthInpaint * 1 / fx
                    y3 = (y - cy) * depthInpaint * 1 / fy
                    z3 = depthInpaint

                    x3 = np.reshape(x3, len(x3) * len(x3[0]))
                    y3 = np.reshape(y3, len(y3) * len(y3[0]))
                    z3 = np.reshape(z3, len(z3) * len(z3[0]))
                    pointsMat = np.vstack((x3, -y3, z3))


                    corners = np.zeros((8,3))
                    basis_ori = items[0]

                    label = items[3][0]
                    print("label : " , label)
                    inds = np.argsort(-abs(items[0][:, 0]))

                    basis = items[0][inds, :]
                    coeffs = items[1][0, inds]

                    inds = np.argsort(-abs(basis[1:, 1]))

                    centroid = items[2]

                    basis = flip_towards_viewer(basis, np.matlib.repmat(centroid, 3, 1))
                    coeffs = abs(coeffs)

                    orientation = items[6][0]

                    corners[0, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
                    corners[1, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
                    corners[2, :] = basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
                    corners[3, :] = -basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]


                    corners[4, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
                    corners[5, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
                    corners[6, :] = basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
                    corners[7, :] = -basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]

                    corners += np.matlib.repmat(centroid, 8, 1)

                    #radian값.
                    theta = math.atan2(orientation[1], orientation[0])
                    theta_s = str(theta*180.0/math.pi)[:6]
                    #degree값
                    dtheta = math.degrees(theta)
                    #x좌표 : length , y좌표 : width, z좌표 : height
                    label_length = corners[0, 0] - corners[1, 0] # 0 - 1
                    label_height = corners[2, 1] - corners[1, 1] # 2 - 1
                    label_width = corners[0, 2] - corners[4, 2] # 0 - 4
                    #project_to_image
                    #new_K[0][3] = 44
                    # cv2.line(color_raw, (int(corners[0, 0]), int(corners[1, 0])),
                    #          (int(corners[0, 1]), int(corners[1, 1])), (0, 0, 255), 2,
                    #          lineType=cv2.LINE_AA)
                    corners = Rtilt @ corners.transpose(1,0)
                    corners = corners.transpose(1,0)
                    corners = corners.astype('float32')
                    sun_permutation = [0,2,1]
                    i= np.argsort(sun_permutation)
                    corners = corners[:, i]

                    pts_3d_homo = np.concatenate(
                        [corners, np.ones((corners.shape[0], 1), dtype=np.float32)], axis=1)
                    pts_2d = np.dot(new_K, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
                    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]

                    face_idx = [[0, 1, 5, 4],  # 앞
                                [1, 2, 6, 5],  # 왼
                                [2, 3, 7, 6],  # 뒤
                                [3, 0, 4, 7]]  # 오
                    for ind_f in range(3, -1, -1):
                        f = face_idx[ind_f]
                        for j in range(4):
                            cv2.line(color_raw, (int(pts_2d[f[j], 0]), int(pts_2d[f[j], 1])),
                                     (int(pts_2d[f[(j + 1) % 4], 0]), int(pts_2d[f[(j + 1) % 4], 1])), (0, 0, 255), 2,
                                     lineType=cv2.LINE_AA)
                    cv2.imshow("img", color_raw)
                    cv2.waitKey()


                    # pts_3d_homo = np.concatenate(
                    #     [corners, np.ones((corners.shape[0], 1), dtype=np.float32)], axis=1)
                    # pts_3d_homo = np.dot(new_R, pts_3d_homo.transpose(1, 0))
                    # pts_2d = np.dot(new_K, pts_3d_homo).transpose(1, 0)
                    # pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]


                    label_center_x = corners[2, 0] + label_length / 2
                    label_center_y = corners[2, 1] - label_height / 2
                    label_center_z = corners[2, 2] - label_width / 2


                    # corners[:, 0] = (corners[:, 0] - cx) * depthInpaint * 1 / fx
                    # corners[:, 1] = (corners[:, 1] - cy) * depthInpaint * 1 / fy
                    # corners[:, 2] = depthInpaint

                    #
                    # label_list = [label, centroid[0], centroid[1], centroid[2], label_length, label_width, label_height, theta]
                    # label_lists.append(label_list[:])
                    #label_list.clear()
                    ax.plot([corners[0, 0], corners[1, 0]], [corners[0, 1], corners[1, 1]],
                            zs=[corners[0, 2], corners[1, 2]], c='r')

                    ax.plot([corners[1, 0], corners[2, 0]], [corners[1, 1], corners[2, 1]],
                            zs=[corners[1, 2], corners[2, 2]], c='r')
                    ax.plot([corners[2, 0], corners[3, 0]], [corners[2, 1], corners[3, 1]],
                            zs=[corners[2, 2], corners[3, 2]], c='r')
                    ax.plot([corners[3, 0], corners[0, 0]], [corners[3, 1], corners[0, 1]],
                            zs=[corners[3, 2], corners[0, 2]], c='r')
                    ax.plot([corners[4, 0], corners[5, 0]], [corners[4, 1], corners[5, 1]],
                            zs=[corners[4, 2], corners[5, 2]], c='r')
                    ax.plot([corners[5, 0], corners[6, 0]], [corners[5, 1], corners[6, 1]],
                            zs=[corners[5, 2], corners[6, 2]], c='r')
                    ax.plot([corners[6, 0], corners[7, 0]], [corners[6, 1], corners[7, 1]],
                            zs=[corners[6, 2], corners[7, 2]], c='r')
                    ax.plot([corners[7, 0], corners[4, 0]], [corners[7, 1], corners[4, 1]],
                            zs=[corners[7, 2], corners[4, 2]], c='r')

                    ax.plot([corners[0, 0], corners[4, 0]], [corners[0, 1], corners[4, 1]],
                            zs=[corners[0, 2], corners[4, 2]], c='r')
                    ax.plot([corners[1, 0], corners[5, 0]], [corners[1, 1], corners[5, 1]],
                            zs=[corners[1, 2], corners[5, 2]], c='r')
                    ax.plot([corners[2, 0], corners[6, 0]], [corners[2, 1], corners[6, 1]],
                            zs=[corners[2, 2], corners[6, 2]], c='r')
                    ax.plot([corners[3, 0], corners[7, 0]], [corners[3, 1], corners[7, 1]],
                            zs=[corners[3, 2], corners[7, 2]], c='r')

                    ax.text3D(corners[0,0], corners[0,1], corners[0,2], label, fontsize=10)
                    # ax.text3D(corners[0,0], corners[0,1], corners[0,2], label+" / "+theta_s, fontsize=10, color='blue')
                    pass
                #label_img_path = data[1][0][i][5]
                #print(label_img_path)

                label_lists.clear()
                pass

            bgr = np.zeros((len(rgb),3))
            bgr[:,0] = rgb[:,2]
            bgr[:,1] = rgb[:,1]
            bgr[:,2] = rgb[:,0]
            ax.scatter(x3, z3, y3, c=bgr, depthshade=False)
            pyplot.show()

        else:
            continue
        pass
            #(2) depth를 m단위로 전환

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
