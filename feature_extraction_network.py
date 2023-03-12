import argparse
import sys
from statistics import mode

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications import ResNet50, Xception, VGG16, InceptionV3, MobileNet, DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.models import Model
from PIL import Image
import configparser
import mysql.connector
from sqlalchemy import create_engine
import gzip

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetname', type=str, help='dataset name', required=True)
    parser.add_argument('--architecture', choices=['resnet50', 'xception', 'vgg16', 'inceptionv3', 'mobilenet', 'densenet121'], help='cnn network architecture', required=True)
    parser.add_argument('--features', type=int, help='number of features', required=True, default=1024)
    parser.add_argument('--outputdirectory', type=str, help='output directory to store the features', required=True)

    return parser.parse_args(argv)

def create_model(architecture, input_shape, num_features):
    # Input layer
    input_layer = Input(shape=input_shape)

    # Reshape motion_residual to have 3 channels
    x = tf.keras.backend.stack((input_layer,)*3, axis=-1)

    # Load pre-trained ResNet50 without the final classification layer
    if architecture == 'resnet50':
        base_model = ResNet50(include_top=False, input_tensor=x, weights='imagenet')
    elif architecture == 'xception':
        base_model = Xception(include_top=False, input_tensor=x, weights='imagenet')
    elif architecture == 'vgg16':
        base_model = VGG16(include_top=False, input_tensor=x, weights='imagenet')
    elif architecture == 'inceptionv3':
        base_model = InceptionV3(include_top=False, input_tensor=x, weights='imagenet')
    elif architecture == 'mobilenet':
        base_model = MobileNet(include_top=False, input_tensor=x, weights='imagenet')
    elif architecture == 'densenet121':
        base_model = DenseNet121(include_top=False, input_tensor=x, weights='imagenet')

    # Freeze the weights of the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add new classification layers
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    output_layer = Dense(num_features)(x)  # remove activation function

    # Define the model with input and output layers
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

def main(argv):
    args = parse_args(argv)

    dataset_name = args.datasetname
    architecture = args.architecture
    num_features = args.features
    output_directory = args.outputdirectory

    # Read the MySQL connection details from config.ini
    config = configparser.ConfigParser()
    config.read('config.ini')
    user = config['mysql']['user']
    password = config['mysql']['password']
    host = config['mysql']['host']
    database = config['mysql']['database']

    #    # Connect to MySQL database
    mydb = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )

    input_shape = (224, 224)

    model = create_model(architecture, input_shape, num_features)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    # Set up cursor to execute SQL queries
    mycursor = mydb.cursor()

    # Execute SQL query to retrieve data
    sql = "SELECT * FROM `deepfake-video-detection`.`celeb-df-v2-motion-residual`"
    mycursor.execute(sql)

    # Fetch the data as a list of tuples
    data = mycursor.fetchall()

    # Convert the data to a DataFrame
    flow_df = pd.DataFrame(data, columns=['index', 'video_name', 'frame_name', 'motion_residual', 'label'])
    # print(flow_df.head())

    # Close the database connection
    mydb.close()

    # Group the flow_df by video_name
    grouped_df = flow_df.groupby('video_name')

    # Load motion_residual data and labels from each video
    motion_residual_data = []
    labels = []
    for video_name, group in grouped_df:
        motion_residuals = []
        label = group['label'].iloc[0] # get the label of the video
        current_label = []
        for i, row in group.iterrows():
            motion_residual = row['motion_residual']
            motion_residual = np.frombuffer(motion_residual, dtype=np.uint8).reshape((224, 224))
            motion_residual = Image.fromarray(motion_residual)
            motion_residual = motion_residual.resize(input_shape, resample=Image.BICUBIC)
            motion_residual = np.array(motion_residual)
            motion_residuals.append(motion_residual)
            current_label.append(label) # add label for each frame in the video
        if len(motion_residuals) < 300:
            # add padding to motion_residuals
            num_frames_to_pad = 300 - len(motion_residuals)
            padding = np.zeros((num_frames_to_pad, input_shape[0], input_shape[1]))
            motion_residuals += padding.tolist()
            current_label += [label] * num_frames_to_pad
        elif len(motion_residuals) > 300:
            motion_residuals = motion_residuals[:300]
            current_label = current_label[:300] # truncate labels for excess frames
        labels.append(mode(current_label))
        motion_residual_data.append(np.array(motion_residuals))
        # Reset motion_residuals and labels lists for next video

    motion_residual_data = np.array(motion_residual_data)

    # Reshape the motion_residual_data from (None, 300, 224, 224) to (None*30, 224, 224)
    motion_residual_data_reshaped = motion_residual_data.reshape((-1, 224, 224))

    # Normalize the data to have values between 0 and 1
    motion_residual_data_reshaped = motion_residual_data_reshaped / 255.0

    print(motion_residual_data_reshaped.shape)
    
    # Get feature vectors
    features = model.predict(motion_residual_data_reshaped)

    # Reshape the features from (270, 1024) to (9, 30, features)
    features_reshaped = features.reshape((-1, 300, num_features))

    # Print the shapes of the resulting arrays
    print(features_reshaped.shape)

    # Convert label list to numpy array
    labels = np.array(labels)

    # Save features
    np.save(output_directory + dataset_name + '_' + architecture + '_' + str(num_features) +'.npy', features_reshaped)

    # Save Labels
    np.save(output_directory + dataset_name + '_' + architecture + '_' + str(num_features) + '_' + 'labels' +'.npy', labels)

if __name__ == '__main__':
    main(sys.argv[1:])