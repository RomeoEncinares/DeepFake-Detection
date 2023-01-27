import json
import pandas as pd
import os

directory = "dataset_processed/train/"

# load the JSON file
with open('dataset/train_sample_videos/metadata.json') as json_file:
    videos = json.load(json_file)

# create a list to hold the data for the dataframe
data = []

# loop through the videos and extract the frames
for video_name, video_info in videos.items():
    label = video_info['label']
    split = video_info['split']
    original = video_info['original']
#     print(video_name.rsplit('.', 1)[0])
    for frame_name in os.listdir(directory):
        if video_name.rsplit('.', 1)[0] in frame_name:
             data.append({'video_name': video_name,'frame_name': frame_name, 'label': label, 'split': split})
        
df = pd.DataFrame(data, columns=['video_name', 'frame_name', 'label', 'split'])
df.to_csv('df.csv')