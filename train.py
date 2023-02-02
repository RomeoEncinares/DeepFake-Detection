import numpy as np
import pandas as pd
import keras.utils as image
from tensorflow import keras
from keras.applications.resnet import preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense

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
        x = preprocess_input(x)
        x_train.append(x)
        y_train.append(row['label'])

x_train = np.array(x_train)
y_train = np.array(y_train)

# Load the model
base_model = keras.applications.ResNet50(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3))

# Freeze the layers of the model
for layer in base_model.layers:
    layer.trainable = False

# Add a new layer for feature extraction
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
features = Dense(256, activation='relu')(x)
feature_extraction_network = keras.models.Model(inputs=base_model.input, outputs=features)