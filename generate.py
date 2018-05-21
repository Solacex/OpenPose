
# coding: utf-8

import cv2 as cv 
import numpy as np
import scipy
import PIL.Image
import math
import caffe
import time
from config_reader import config_reader
import util
import copy
import os
import scipy
import matplotlib
import pylab as plt
from scipy.ndimage.filters import gaussian_filter


# In[2]:

#----------------initial model-----------------
param, model = config_reader()

if param['use_gpu']: 
    caffe.set_mode_gpu()
    caffe.set_device(param['GPUdeviceNumber']) # set to your device!
else:
    caffe.set_mode_cpu()
net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)


# In[3]:

def extract_feat(model, oriImg, param):
    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    # first figure shows padded images
    multiplier = [x * model['boxsize'] / oriImg.shape[0] for x in param['scale_search']]

    # We use 1080 (8G memory), so we set 3 scales here.
    for m in range(3): #range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model['stride'], model['padValue'])
        #print imageToTest_padded.shape

        net.blobs['data'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
        #net.forward() # dry run
        net.blobs['data'].data[...] = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;
        #start_time = time.time()
        output_blobs = net.forward()
        #print('At scale %d, The CNN took %.2f ms.' % (m, 1000 * (time.time() - start_time)))

        # extract outputs, resize, and remove padding
        heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1,2,0)) # output 1 is heatmaps
        heatmap = cv.resize(heatmap, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        heatmap = cv.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)
    
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        
    all_peaks = []
    peak_counter = 0

    for part in range(19-1):
        x_list = []
        y_list = []
        map_ori = heatmap_avg[:,:,part]
        map = gaussian_filter(map_ori, sigma=3)
    
        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]
    
        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)
    return all_peaks, heatmap_avg


# In[4]:

def keep_one(peaks):
    for i in range(18):
        if len(peaks[i])>=2:
            del peaks[i][1:]
    return peaks


# In[9]:

test_dir = '/home/liguangrui/HWP_key3/train/'
list_dir = '/home/liguangrui/HWP_key3/train.list'

#/1631_c5_f0126448.jpg'
count = 0;
points = {}
heatmaps = {}
file = open(list_dir)
img_id=0
while 1:
    line = file.readline()
    if not line:
        break
    name = line[:43]
    img_path = os.path.join(test_dir,name)
    oriImg = cv.imread(img_path) 
    if oriImg is None:
        continue
    peak18, heatmap = extract_feat(model, oriImg, param)
    peak18 = keep_one(peak18)
    points[count] = peak18
#    heatmaps[count] = heatmap
    count = count+1;
    print(count,name)
        #oriImg.release()



#for root, dirs, files in os.walk(test_dir):
 #   for name in files:
  #      img_id = name[:-4]


#------------save points-------------
import json
with open('./result/hwp3_dot.json','w') as fp:
    json.dump(points, fp, sort_keys=True, indent=2)
    
#------------save heatmaps------------
#import pickle
#with open('./result/gallery_heatmaps.pickle','wb') as handle:
#    pickle.dump(heatmaps, handle, pickle.HIGHEST_PROTOCOL)

