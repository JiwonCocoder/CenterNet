import os
import numpy as np

f = open("C:\\obj_detection\\darknet-master\\darknet-master\\build\\darknet\\x64\\data\\5k.txt", 'r')
f_write = open('data/val2014.list','w')
lines = f.readlines()
x=[]
for line in lines:
    x.append('images/val2014/'+line[:-1].split('/')[-1])
f_write.write('\n'.join(x))
