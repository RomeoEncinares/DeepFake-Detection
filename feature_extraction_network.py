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


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetname', type=str, help='dataset name', required=True)
    parser.add_argument('--dataframe', type=str, help='csv dataframe', required=True)
    parser.add_argument('--opticalflow', type=str, help='csv opticalflow dataframe', required=False)
    parser.add_argument('--opticalflowoutput', type=str, help='csv opticalflow dataframe output', required=False)
    parser.add_argument('--architecture', choices=['resnet50', 'xception', 'vgg16', 'inceptionv3', 'mobilenet', 'densenet121'], help='cnn network architecture', required=True)
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

        print('{}: {}'.format(next_row.video_name, next_row.frame_name))

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
    
    input_shape = (224, 224)

    model = create_model(architecture, input_shape, num_features)
    model.compile(loss='binary_crossentropy', optimizer='adam')

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

    # Reshape the features from (270, 1024) to (9, 30, 1024)
    features_reshaped = features.reshape((-1, 300, 1024))

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