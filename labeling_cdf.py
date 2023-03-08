import argparse
import os
import re
import sys

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetRoot', type=str, help='Dataset root directory', required=True)
    parser.add_argument('--dataframeOutput', type=str, help='CSV dataframe output directory', required=True)

    return parser.parse_args(argv)

def sort_key(filename):
    # Extract the numbers within the parentheses using regular expression
    match = re.search(r'\((\d+)\)-(\d+)', filename)
    if match:
        group1 = int(match.group(1))
        group2 = int(match.group(2))
        return group1, group2
    else:
        return filename

def label(frames_list, current_directory, dataframe_dir):
    data = []
    for frame in frames_list:
        video_name = frame.rsplit('-', 1)[0]
        frame_name = frame
        file_path = current_directory + frame
        label = 'DEEPFAKE' if current_directory.split('/', 2)[2].rstrip('/') == 'Celeb-synthesis' else 'REAL'
        data.append({'video_name': video_name,'frame_name': frame_name, 'file_path': file_path,'label': label})

    df = pd.DataFrame(data, columns=['video_name', 'frame_name', 'file_path', 'label'])
    df.to_csv(dataframe_dir + 'df.csv')

    return df

def main(argv):
    args = parse_args(argv)

    dataset_dir = args.datasetRoot
    dataframe_dir = args.dataframeOutput
    dataset_dir_sub_root = ['Celeb-real', 'Celeb-synthesis', 'Youtube-real']

    le = LabelEncoder()
    for i in dataset_dir_sub_root:
        current_directory = dataset_dir + i + '/'
        frames_list = os.listdir(dataset_dir + i + '/')
        if i == 'Celeb-real':
            frames_list = sorted(frames_list, key=sort_key)
            df1 = label(frames_list, current_directory, dataframe_dir)
        elif i == 'Celeb-synthesis':
            frames_list = sorted(frames_list, key=sort_key)
            df2 = label(frames_list, current_directory, dataframe_dir)
        elif i == 'Youtube-real':
            frames_list = sorted(frames_list, key=sort_key)
            df3 = label(frames_list, current_directory, dataframe_dir)

    df_combined = pd.concat([df1, df2, df3])
    
    df_combined.label = le.fit_transform(df_combined.label)
    df_combined.to_csv(dataframe_dir + 'df.csv')

if __name__ == '__main__':
    main(sys.argv[1:])