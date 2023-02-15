import argparse
import os
import sys

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetRoot', type=str, help='Dataset root directory', required=True)
    parser.add_argument('--dataframeOutput', type=str, help='CSV dataframe output directory', required=True)

    return parser.parse_args(argv)

def sort_key_celeb_real(filename):
    id_, num_ext = filename.split('_')
    num, ext = num_ext.split('.')
    num_parts = num.split('-')
    return (int(id_[2:]), int(num_parts[0]), int(num_parts[1]))

def sort_key_celeb_synthesis(filename):
    parts = filename.split('_')
    id_values = [int(part[2:]) for part in parts if part.startswith('id')]
    num_ext = [part for part in parts if '-' in part][0]
    num_parts = num_ext.split('-')
    return tuple(id_values + [int(num_parts[0]), int(num_parts[1].split('.')[0])])

def sort_key_youtube_real(filename):
    parts = filename.split('-')
    return (int(parts[0]), int(parts[1].split('.')[0]))

def label(frames_list, current_directory, dataframe_dir):
    data = []
    for frame in frames_list:
        video_name = frame.rsplit('-', 1)[0]
        frame_name = frame
        file_path = current_directory + frame
        label = 'FAKE' if current_directory.split('/', 2)[1] == 'Celeb-synthesis' else 'ORIGINAL'
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
            frames_list = sorted(frames_list, key=sort_key_celeb_real)
            df1 = label(frames_list, current_directory, dataframe_dir)
        elif i == 'Celeb-synthesis':
            frames_list = sorted(frames_list, key=sort_key_celeb_synthesis)
            df2 = label(frames_list, current_directory, dataframe_dir)
        elif i == 'Youtube-real':
            frames_list = sorted(frames_list, key=sort_key_youtube_real)
            df3 = label(frames_list, current_directory, dataframe_dir)

    df_combined = pd.concat([df1, df2, df3])

    df_combined.label = le.fit_transform(df_combined.label)
    df_combined.to_csv(dataframe_dir + 'df.csv')

if __name__ == '__main__':
    main(sys.argv[1:])