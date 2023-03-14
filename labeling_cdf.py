import argparse
import os
import re
import sys

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import configparser
import mysql.connector
from sqlalchemy import create_engine


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetRoot', type=str, help='Dataset root directory', required=True)
    parser.add_argument('--tablename', type=str, help='table name', required=True)

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

def label(frames_list, current_directory):
    data = []
    for frame in frames_list:
        video_name = frame.rsplit('-', 1)[0]
        frame_name = frame
        file_path = current_directory + frame
        label = 'DEEPFAKE' if current_directory.split('/', 2)[2].rstrip('/') == 'Celeb-synthesis' else 'REAL'
        data.append({'video_name': video_name,'frame_name': frame_name, 'file_path': file_path,'label': label})

    df = pd.DataFrame(data, columns=['video_name', 'frame_name', 'file_path', 'label'])

    return df

def main(argv):
    args = parse_args(argv)

    dataset_dir = args.datasetRoot
    table_name = args.tablename
    dataset_dir_sub_root = ['Celeb-real', 'Celeb-synthesis', 'Youtube-real']

    le = LabelEncoder()
    for i in dataset_dir_sub_root:
        current_directory = dataset_dir + i + '/'
        frames_list = os.listdir(dataset_dir + i + '/')
        if i == 'Celeb-real':
            frames_list = sorted(frames_list, key=sort_key)
            df1 = label(frames_list, current_directory)
        elif i == 'Celeb-synthesis':
            frames_list = sorted(frames_list, key=sort_key)
            df2 = label(frames_list, current_directory)
        elif i == 'Youtube-real':
            frames_list = sorted(frames_list, key=sort_key)
            df3 = label(frames_list, current_directory)

    df_combined = pd.concat([df1, df2, df3])
    df_combined.label = le.fit_transform(df_combined.label)

    # Read the MySQL connection details from config.ini
    config = configparser.ConfigParser()
    config.read('config.ini')
    user = config['mysql']['user']
    password = config['mysql']['password']
    host = config['mysql']['host']
    database = config['mysql']['database']
    
    # Store credentials
    engine = create_engine(f'mysql://{user}:{password}@{host}/{database}')

    # Connect to MySQL database
    cnx = mysql.connector.connect(user=user, password=password, host=host, database=database)
    cursor = cnx.cursor()

    df_combined.to_sql(name=table_name, con=engine, if_exists='append', index=False)

    # Close the database connection
    cursor.close()
    cnx.close()

if __name__ == '__main__':
    main(sys.argv[1:])