import argparse
import configparser
import sys

import cv2
import mysql.connector
import pandas as pd


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, help='start at index', required=False)
    parser.add_argument('--end', type=int, help='end at index', required=False)
    
    return parser.parse_args(argv)

def compute_optical_flow(start_index, end_index):
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

    return pd.DataFrame(flow_data)

def insert_motion_residual(mydb, insert_values):
    insert_query = """
    INSERT INTO `deepfake-video-detection`.`celeb-df-v2-motion-residual` (video_name, frame_name, motion_residual, label)
    VALUES (%s, %s, %s, %s)
    """
    cursor = mydb.cursor()
    cursor.execute(insert_query, insert_values)
    mydb.commit()


def main(argv):
    args = parse_args(argv)

    start_index = args.start
    end_index = args.end

    # Read the MySQL connection details from config.ini
    config = configparser.ConfigParser()
    config.read('config.ini')
    user = config['mysql']['user']
    password = config['mysql']['password']
    host = config['mysql']['host']
    database = config['mysql']['database']

    # Connect to MySQL database
    mydb = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )

    # Set up cursor to execute SQL queries
    mycursor = mydb.cursor()

    # Execute SQL query to retrieve data
    sql = "SELECT * FROM `deepfake-video-detection`.`celeb-df-v2-raw`"
    mycursor.execute(sql)

    # Fetch the data as a list of tuples
    data = mycursor.fetchall()

    # Convert the data to a DataFrame
    df = pd.DataFrame(data, columns=['index', 'video_name', 'frame_name', 'file_path', 'label'])
    # print(df.head())

if __name__ == '__main__':
    main(sys.argv[1:])