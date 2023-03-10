import argparse
import sys

import cv2
import pandas as pd


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataframe', type=str, help='csv dataframe', required=True)
    parser.add_argument('--opticalflow', type=str, help='csv opticalflow dataframe', required=False)
    parser.add_argument('--opticalflowoutput', type=str, help='csv opticalflow dataframe output', required=False)
    parser.add_argument('--start', type=int, help='start at index', required=False)
    parser.add_argument('--end', type=int, help='end at index', required=False)
    
    return parser.parse_args(argv)

def compute_optical_flow(df, start_index, end_index):
    flow_data = []
    for i in range(start_index, end_index):
        row = df.iloc[i]
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

def main(argv):
    args = parse_args(argv)

    df_directory = args.dataframe
    df_flow_directory = args.opticalflow
    df_flow_output_directory = args.opticalflowoutput
    start_index = args.start
    end_index = args.end

    df = pd.read_csv(df_directory)
    
    if df_flow_directory != None:
        # Read the existing data from file
        flow_df = pd.read_csv(df_flow_directory, index_col=0)
        
        # Compute the new flow data and append it to the existing data
        new_flow_df = compute_optical_flow(df, start_index, end_index)
        new_flow_df.index = range(flow_df.index.max()+1, flow_df.index.max()+1+len(new_flow_df))
        flow_df = pd.concat([flow_df, new_flow_df], ignore_index=True)
        
        # Write the updated DataFrame to file
        flow_df.to_csv(df_flow_output_directory + 'flow_df.csv', index=True)
    else:
        flow_df = compute_optical_flow(df, start_index, end_index)
        flow_df.to_csv(df_flow_output_directory + 'flow_df.csv', index=True)


if __name__ == '__main__':
    main(sys.argv[1:])