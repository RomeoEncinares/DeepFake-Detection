import argparse
import os
import sys
import torch

from facenet_pytorch import MTCNN
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
detector = MTCNN(select_largest=False, post_process=False, device=device)

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='Videos root directory', required=True)
    parser.add_argument('--target', type=str, help='Output root directory', required=True)
    parser.add_argument('--frameRate', type=float, help='Frames per video', default=0.10)
    parser.add_argument('--startVideo', type=str, help='Start extraction on name video')
    parser.add_argument('--endVideo', type=str, help='End extraction on name video')

    return parser.parse_args(argv)

def sort_key_video(filename):
    id_, num_ext = filename.split(' (')
    num, _ = num_ext.split(').')
    return int(num)

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
    video_list = sorted(os.listdir(source_dir), key=sort_key_video)
    video_list_count = len(video_list)
    start_video_index = 0
    
    if start_video != None:
        start_video_index = video_list.index(start_video)

    if end_video != None:
        video_list_count = video_list.index(end_video) + 1

    for video in range(start_video_index, video_list_count):
        getFrame(source_dir, target_dir, frame_rate, video_list[video])

def getFrame(source_dir, target_dir, frame_rate, video):
    video_path = source_dir + video
    vidcap = cv2.VideoCapture(video_path)

    def saveFrame(sec, target_dir):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            try:
                process(target_dir, video, image, count)
            except AttributeError:
                pass
        return hasFrames
        
    sec = 0
    frameRate = frame_rate # Capture image in each x second
    count = 1
    success = saveFrame(sec, target_dir)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = saveFrame(sec, target_dir)

def process(target_dir, frame_name, frame, count):
    frame_read = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    save_paths = target_dir + frame_name.rsplit('.', 1)[0] + "-" + str(count) + ".jpg"
    detector(frame_read, save_paths)

if __name__ == '__main__':
    main(sys.argv[1:])