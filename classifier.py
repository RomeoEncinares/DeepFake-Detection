import argparse
import sys

import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Input
from sklearn.model_selection import train_test_split

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, help='features', required=True)
    parser.add_argument('--labels', type=str, help='labels', required=True)
    parser.add_argument('--outputdirectory', type=str, help='output directory to store the models', required=True)

    return parser.parse_args(argv)

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask
    
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