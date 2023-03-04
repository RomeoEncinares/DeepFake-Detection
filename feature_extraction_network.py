import argparse
import sys
from statistics import mode

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.models import Model
from PIL import Image


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataframe', type=str, help='csv dataframe', required=True)
    parser.add_argument('--opticalflow', type=str, help='csv opticalflow dataframe', required=False)
    parser.add_argument('--opticalflowoutput', type=str, help='csv opticalflow dataframe output', required=False)
    parser.add_argument('--architecture', choices=['resnet50'], help='cnn network architecture', required=True)
    parser.add_argument('--features', type=int, help='number of features', required=True, default=1024)
    parser.add_argument('--outputdirectory', type=str, help='output directory to store the features', required=True)

    return parser.parse_args(argv)

def compute_optical_flow(df):
    flow_data = []
    for i, row in df.iterrows():
        # Check if the next frame is from the same video
        next_row = df.iloc[i + 1] if i + 1 < len(df) else None
        if next_row is not None and row['video_name'] != next_row['video_name']:
            continue

        # Read the current frame and the next frame
        frame1 = cv2.imread(row['file_path'])
        if next_row is not None:
            frame2 = cv2.imread(next_row['file_path'])
        else:
            # If the next frame is from a different video, skip computing optical flow
            continue

        frame1 = cv2.resize(frame1, (224, 224), interpolation=cv2.INTER_LINEAR)
        frame2 = cv2.resize(frame2, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Convert the frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Compute the optical flow between the frames
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Convert flow vectors to polar coordinates
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Threshold the magnitude to create a binary mask of moving regions
        magnitude_threshold = 4
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        ret, mask = cv2.threshold(magnitude, magnitude_threshold, 1, cv2.THRESH_BINARY)

        # Apply the mask to the second frame to highlight the moving regions
        frame2_masked = cv2.bitwise_and(frame2, frame2, mask=mask)

        # Subtract the masked second frame from the first frame to create a difference image
        diff = cv2.absdiff(frame1, frame2_masked)
        # Convert difference image to colored
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # diff_rgb = cv2.merge((diff_gray, diff_gray, diff_gray))

        # Store the optical flow data for the current pair of frames
        flow_data.append({
            'video_name': row['video_name'],
            'frame_name': row['frame_name'],
            'motion_residual': diff,
            'label': row['label'],
        })
    return pd.DataFrame(flow_data)

def architecture_resnet50(architecture, input_shape, num_features):
    # Input layer
    input_layer = Input(shape=input_shape)

    # Reshape motion_residual to have 3 channels
    x = tf.keras.backend.stack((input_layer,)*3, axis=-1)

    # Load pre-trained ResNet50 without the final classification layer
    if architecture == 'resnet50':
        base_model = ResNet50(include_top=False, input_tensor=x, weights='imagenet')

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

    df_directory = args.dataframe
    df_flow_directory = args.opticalflow
    df_flow_output_directory = args.opticalflowoutput
    architecture = args.architecture
    num_features = args.features
    output_directory = args.outputdirectory

    if df_flow_directory != None:
        flow_df = pd.read_csv(df_flow_directory)
    else:
        df = pd.read_csv(df_directory)
        flow_df = compute_optical_flow(df)
        flow_df.to_csv(df_flow_output_directory + 'flow_df.csv')

if __name__ == '__main__':
    main(sys.argv[1:])