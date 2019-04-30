# coding: utf-8
import numpy as np
import cv2
import argparse
import sys
from PIL import Image
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import os,sys
from os import listdir
from os.path import isfile, join
import time as t
import csv
import matplotlib.pyplot as plt
import math
#import easydict
from xml.dom import minidom
import json
from pprint import pprint
'''
proj_path = os.getcwd()
mypath = os.path.join(proj_path,'train')
jsonpath = os.path.join(proj_path,'avaiable_json2')
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
jsonfiles = [f for f in listdir(mypath) if isfile(join(jsonpath, f))]
for inputName in onlyfiles:
    print('Processing image: ',inputName)
    filename, file_extension = os.path.splitext(inputName)
    if file_extension == '.png':
        if filename in jsonfiles:
'''

data_path = '/home/yunliang/Data/YunDisk/data_bcidr/Images'
save_path_train = '/home/yunliang/Data/YunDisk/data_bcidr/train/'
save_path_test = '/home/yunliang/Data/YunDisk/data_bcidr/test/'
'''
with open('/home/yunliang/Data/YunDisk/data_bcidr/train_annotation.json', 'r') as f:
    json_file = json.load(f)
    #pprint(json_file)
    #import pdb;pdb.set_trace()
    #imglist = len(json_file)
    for img_name in json_file.keys():
        print('Processing: ':img_name)
        img = cv2.imread(os.path.join(data_path,img_name+'.png'))
        if not os.path.exists(save_path_train):
            os.makedirs(save_path_train)
        cv2.imwrite(os.path.join(save_path_train,img_name+'.png'), img)

'''
with open('/home/yunliang/Data/YunDisk/data_bcidr/test_annotation.json', 'r') as f:
    json_file = json.load(f)
    #pprint(json_file)
    #import pdb;pdb.set_trace()
    #imglist = len(json_file)
    for img_name in json_file.keys():
        #print('Processing: ':img_name)
        img = cv2.imread(os.path.join(data_path,img_name+'.png'))
        if not os.path.exists(save_path_test):
            os.makedirs(save_path_test)
        cv2.imwrite(os.path.join(save_path_test,img_name+'.png'), img)
