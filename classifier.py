import argparse
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.layers import Input
from sklearn.model_selection import train_test_split
from tensorflow import keras


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, help='features', required=True)
    parser.add_argument('--labels', type=str, help='labels', required=True)
    parser.add_argument('--outputdirectory', type=str, help='output directory to store the models', required=True)
    parser.add_argument('--epochs', type=int, help='number of epochs', required=True)

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

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

def vision_transformer_classifier():
    sequence_length = 30
    embed_dim = 1024
    dense_dim = 4
    num_heads = 1
    classes = 2

    inputs = keras.Input(shape=(None, None))
    x = PositionalEmbedding(
        sequence_length, embed_dim, name="frame_position_embedding"
    )(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model

def run_experiment(output_directory, train_data, train_labels, test_data, test_labels, num_epochs):
    filepath = output_directory
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    model = vision_transformer_classifier()
    history = model.fit(
        train_data,
        train_labels,
        validation_split=0.15,
        epochs=num_epochs,
        callbacks=[checkpoint],
    )

    model.load_weights(filepath)
    _, accuracy = model.evaluate(test_data, test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return model
    
def main(argv):
    args = parse_args(argv)

    features = args.features
    labels = args.labels
    output_directory = args.outputdirectory
    num_epochs = args.epochs
    
    features = np.load(features)
    labels = np.load(labels)

    train_data, test_data, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Print the shapes of the resulting arrays
    print(f"Frame features in train set: {train_data.shape}")
    print(f"Frame labels in train set: {train_labels.shape}")

    print(f"Frame features in test set: {test_data.shape}")
    print(f"Frame labels in test set: {test_labels.shape}")

    trained_model = run_experiment(output_directory, train_data, train_labels, test_data, test_labels, num_epochs)

if __name__ == '__main__':
    main(sys.argv[1:])