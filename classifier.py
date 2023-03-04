import argparse
import sys

import numpy as np
from sklearn.model_selection import train_test_split

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, help='features', required=True)
    parser.add_argument('--labels', type=str, help='labels', required=True)
    parser.add_argument('--outputdirectory', type=str, help='output directory to store the models', required=True)

    return parser.parse_args(argv)
    
def main(argv):
    args = parse_args(argv)

    features = args.features
    labels = args.labels
    output_directory = args.outputdirectory
    
    features = np.load(features)
    labels = np.load(labels)

    train_data, test_data, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Print the shapes of the resulting arrays
    print(f"Frame features in train set: {train_data.shape}")
    print(f"Frame labels in train set: {train_labels.shape}")

    print(f"Frame features in test set: {test_data.shape}")
    print(f"Frame labels in test set: {test_labels.shape}")

if __name__ == '__main__':
    main(sys.argv[1:])