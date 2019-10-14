import numpy as np
#f = open("C:\\obj_detection\\darknet-master\\darknet-master\\build\\darknet\\x64\\hello.txt", 'r')
f = open("C:\\obj_detection\\CenterNet-master\\CenterNet-master\\src\\fps_test.txt", 'r')
lines = f.readlines()
x=[]
for line in lines:
    x.append(float(line[:-1]))
f.close()
print(np.average(x))
