import argparse
import os
import sys

from mtcnn import MTCNN
import numpy as np
import cv2

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='Videos root directory', required=True)
    parser.add_argument('--target', type=str, help='Output root directory', required=True)
    parser.add_argument('--frameRate', type=int, help='Frames per video', default=0.25)
    parser.add_argument('--startVideo', type=int, help='Start extraction on nth video', default=1)
    parser.add_argument('--endVideo', type=int, help='End extraction on nth video')

    return parser.parse_args(argv)

def main(argv):
    args = parse_args(argv)
    
    # Parameters parsing
    # dataset/train_sample_videos/
    # dataset_processed/train/
    source_dir = args.source
    target_dir = args.target
    frame_rate = args.frameRate
    start_video = args.startVideo
    end_video = args.endVideo

    # Video list index
    video_list = os.listdir(source_dir)
    video_list_count = len(video_list)
    
    if end_video != None:
        video_list_count = end_video

    for video in range(start_video-1, video_list_count):
        getFrame(source_dir, target_dir, frame_rate, video_list[video])

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

def getFrame(source_dir, target_dir, frame_rate, video):
    video_path = source_dir + video
    print(video_path)
    vidcap = cv2.VideoCapture(video_path)

    def saveFrame(sec, target_dir):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            try:
                align_crop_resize(video, target_dir, count, align(image)[1])
            except AttributeError:
                pass
        return hasFrames
        
    sec = 1
    frameRate = frame_rate # Capture image in each x second
    count=1
    success = saveFrame(sec, target_dir)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = saveFrame(sec, target_dir)

if __name__ == '__main__':
    main(sys.argv[1:])