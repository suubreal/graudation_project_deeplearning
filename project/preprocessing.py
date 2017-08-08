# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 21:31:28 2017

@author: USER
"""

import os

folder='dataset/images/pushup_1'
if not os.path.exists(folder):
    os.makedirs(folder)

import cv2
vidcap = cv2.VideoCapture('dataset/video/example.avi')
count = 0

while True :
    success , image = vidcap.read()
    if not success :
        break
    cv2.imwrite(os.path.join(folder , "frame{:d}.jpg".format(count)) , image)
    count += 1
print ("{} images are extracted in {}." .format(count , folder))

