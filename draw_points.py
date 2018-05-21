
# coding: utf-8


import cv2 as cv 
import numpy as np
import scipy
import PIL.Image
import os
import scipy
import matplotlib
from IPython import get_ipython
#get_ipython().magic(u'matplotlib inline')
import pylab as plt
import os


ori_dir = '/home/liguangrui/HWP_key3/train/'
save_dir='/home/liguangrui/hwp3_dot/'
list_dir = '/home/liguangrui/HWP_key3/train.list'

#------------load points-------------
import json
with open('./result/hwp3_dot.json','r') as fp:
    points = json.load(fp)
    

img_path = './base_64_128.jpg'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
colors = [[12,12,12],[24,24,24],[36,36,36],[48,48,48],[60,60,60],[72,72,72],[84,84,84],[96,96,96],[108,108,108],[120,120,120],[132,132,132],[144,144,144],[156,156,156],[168,168,168],[180,180,180],[192,192,192],[204,204,204],[216,216,216],[228,228,228],[240,240,240],[252,252,252]]
color= [255,255,255]
#for (k,v) in points.iteritems():
file = open(list_dir)

for (k,v) in points.items():
    peak18 = v
    filename = file.readline()[:43] 
    ori = cv.imread(ori_dir + filename)
    print filename
    w,h = ori.shape[0:2]
    if h>w:
        continue
    # visualize
    #print w,h
    #print peak18[13][:]
    if np.size(peak18)<10:
        continue
    oriImg = cv.imread(img_path)
    oriImg = cv.resize(oriImg,(h,w))
    is_skip = 0
    empty_num=0
  '''  if len(peak18[4])==0 and len(peak18[7])==0:
        continue
    if len(peak18[13])==0 and len(peak18[10])==0:
        continue
    if len(peak18[11])==0 and len(peak18[8])==0:
        continue
    if len(peak18[2])==0 and len(peak18[5])==0:
        continue

    if len(peak18[2])!=0 and peak18[2][0][1]>w/2:
        continue
    if len(peak18[5])!=0 and peak18[5][0][1]>w/2:
        continue 
    if len(peak18[10])!=0 and peak18[10][0][1]<w/2:
        continue 
    if len(peak18[13])!=0 and peak18[13][0][1]<w/2:
        continue 
    if len(peak18[11])!=0 and peak18[11][0][1]<w/3:
        continue 
    if len(peak18[8])!=0 and peak18[8][0][1]<w/3:
        continue '''

        
    for i in range(18):

        if len(peak18[i])==0:
            empty_num+=1
        '''elif peak18[i][0][1]>w or peak18[i][0][0]>h:
            is_skip+=1'''

        for j in range(len(peak18[i])):
            cv.circle(oriImg, tuple(peak18[i][j][0:2]), 4, color, thickness=-1)

    print 18-empty_num
    if empty_num<4 and is_skip ==0:
        cv.imwrite(save_dir+filename, oriImg)
    #plt.imshow(to_plot[:,:,[2,1,0]])



#    if len(peak18[8])>0 and len(peak18[9])>0:
 #       cv.line(oriImg, tuple(peak18[8][0][0:2]),tuple(peak18[9][0][0:2]),colors,3)
  #  if len(peak18[10])>0 and len(peak18[9])>0:
   #     cv.line(oriImg, tuple(peak18[10][0][0:2]),tuple(peak18[9][0][0:2]),colors,3)
    #if len(peak18[11])>0 and len(peak18[12])>0:
     #   cv.line(oriImg, tuple(peak18[11][0][0:2]),tuple(peak18[12][0][0:2]),colors,3)
    #if len(peak18[12])>0 and len(peak18[13])>0:
     #   cv.line(oriImg, tuple(peak18[12][0][0:2]),tuple(peak18[13][0][0:2]),colors,3)
