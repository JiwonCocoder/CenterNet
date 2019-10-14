import numpy as np
def read_clib(calib_path):
    f = open(calib_path, 'r')
    line = f.readlines()
    #line은 하나일 것. 그리고 총 12개의 값이 들어있을 것. 구분은 ' ' 으로 되어 있을 것.
    calib = np.array(line[:-1].split(' ')[0:], dtype=np.float32)
    return calib