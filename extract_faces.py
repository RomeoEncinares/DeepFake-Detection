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
    parser.add_argument('--frameRate', type=int, help='Frames per video', default=300)
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
        print(video_list[video])
        getFrame(source_dir, target_dir, video_list[video], frame_rate)


def getFrame(source_dir, target_dir, video, num_frames=300):
    video_path = os.path.join(source_dir, video)
    vidcap = cv2.VideoCapture(video_path)

    # calculate the total number of frames in the video
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # calculate the duration of the video in seconds
    duration_sec = total_frames / vidcap.get(cv2.CAP_PROP_FPS)

    # calculate the desired frame rate
    frame_rate = num_frames / duration_sec

    # calculate the interval between frames to extract
    interval = int(total_frames // num_frames)

    def saveFrame(count, target_dir):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, (count-1) * interval)
        hasFrames, image = vidcap.read()
        if hasFrames:
            try:
                process(target_dir, video, image, count)
            except AttributeError:
                pass
        return hasFrames

    for count in range(1, num_frames+1):
        success = saveFrame(count, target_dir)
        if not success:
            break

def process(target_dir, frame_name, frame, count):
    frame_read = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    save_paths = target_dir + frame_name.rsplit('.', 1)[0] + "-" + str(count) + ".jpg"
    detector(frame_read, save_paths)

if __name__ == '__main__':
    main(sys.argv[1:])