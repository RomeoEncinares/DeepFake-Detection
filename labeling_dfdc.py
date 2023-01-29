import json
import pandas as pd
import os
import argparse
import sys

from sklearn.preprocessing import LabelEncoder

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, help='JSON metadata directory', required=True)
    parser.add_argument('--datasetRoot', type=str, help='Dataset root directory', required=True)
    parser.add_argument('--dataframeRoot', type=str, help='CSV dataframe output directory', required=True)

    return parser.parse_args(argv)

def main(argv):
    args = parse_args(argv)

    # dataset_processed/
    metadata_dir = args.metadata
    dataset_dir = args.datasetRoot
    dataframe_dir = args.dataframeRoot

    le = LabelEncoder()

    # load the JSON file
    # dataset/train_sample_videos/metadata.json
    with open(metadata_dir) as json_file:
        videos = json.load(json_file)

    # create a list to hold the data for the dataframe
    data = []

    # loop through the videos and extract the frames
    for video_name, video_info in videos.items():
        label = video_info['label']
        split = video_info['split']
        original = video_info['original']
    #     print(video_name.rsplit('.', 1)[0])
        for frame_name in os.listdir(dataset_dir):
            if video_name.rsplit('.', 1)[0] in frame_name:
                data.append({'video_name': video_name,'frame_name': frame_name, 'file_path': dataset_dir + frame_name,'label': label, 'split': split})
            
    df = pd.DataFrame(data, columns=['video_name', 'frame_name', 'file_path', 'label', 'split'])
    df.label = le.fit_transform(df.label)
    df.to_csv(dataframe_dir + 'df.csv')
    # print(dataframe_dir + 'df.csv')

if __name__ == '__main__':
    main(sys.argv[1:])