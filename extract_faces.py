from mtcnn import MTCNN
from scipy import ndimage

import pandas as pd
import cv2
import os
import sys
import random
import math
import numpy as np
import mtcnn
import matplotlib
import matplotlib.pyplot as plt

filepath_source = "dataset/train_sample_videos/"
filepath_target = "dataset_processed/train/"

video_list = os.listdir(filepath_source)

def rotate_bound(image, angle):
    #rotates an image by the degree angle
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1]) 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin)) 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH)) 

def align(img):
    detector = MTCNN()   
    data=detector.detect_faces(img)
    biggest=0
    if data !=[]:
        for faces in data:
            box=faces['box']            
            # calculate the area in the image
            area = box[3]  * box[2]
            if area>biggest:
                biggest=area
                bbox=box                
                keypoints=faces['keypoints']
                left_eye=keypoints['left_eye']
                right_eye=keypoints['right_eye']                 
        lx,ly=left_eye        
        rx,ry=right_eye
        dx=rx-lx
        dy=ry-ly
        tan=dy/dx
        theta=np.arctan(tan)
        theta=np.degrees(theta)    
        img=rotate_bound(img, theta)        
        return (True,img)
    else:
        return (False, None)

def crop_image(img):
    detector = MTCNN()
    data=detector.detect_faces(img)
    biggest=0
    if data !=[]:
        for faces in data:
            box=faces['box']            
            # calculate the area in the image
            area = box[3]  * box[2]
            if area>biggest:
                biggest=area
                bbox=box 
        bbox[0]= 0 if bbox[0]<0 else bbox[0]
        bbox[1]= 0 if bbox[1]<0 else bbox[1]
        img=img[bbox[1]: bbox[1]+bbox[3],bbox[0]: bbox[0]+ bbox[2]]        
        return (True, img) 
    else:
        return (False, None)

def align_crop_resize(video_name, filepath, count, image): 
    img=image # read in the image
    shape=img.shape
    status,img=align(img) # rotates the image for the eyes are horizontal
    if status:                
        cstatus, img=crop_image(img) # crops the aligned image to return the largest face
        if cstatus:
            cv2.imwrite(filepath + video_name.rsplit('.', 1)[0] + "-" + str(count) + ".jpg", img) # Save frame as JPG file # save the image

def getFrame(source, filepath_target, video):
    video_path = source + video
    print(video_path)
    vidcap = cv2.VideoCapture(video_path)

    def saveFrame(sec, filepath):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            try:
#                 print(filepath)
                align_crop_resize(video, filepath, count, align(image)[1])
            except AttributeError:
                pass
        return hasFrames
        
    sec = 1
    frameRate = 0.25 # Capture image in each 0.5 second
    count=1
    success = saveFrame(sec, filepath_target)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = saveFrame(sec, filepath_target)

for video in video_list:
    getFrame(filepath_source, filepath_target, video)