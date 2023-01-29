import numpy as np
import pandas as pd
import tensorflow as tf
import keras.utils as image
from tensorflow import keras
import matplotlib.pyplot as plt

df = pd.read_csv('dataset_processed-Copy/df.csv')

# Create x_train and y_train arrays
x_train = []
y_train = []

for index, row in df.iterrows():
    # Only use rows where "split" column is "train"
    if row['split'] == 'train':
        # Read the image file
        img = image.load_img(row['file_path'], target_size=(224, 224))
        x = image.img_to_array(img)
        x = x/255
        x_train.append(x)
        y_train.append(row['label'])

x_train = np.array(x_train)
y_train = np.array(y_train)