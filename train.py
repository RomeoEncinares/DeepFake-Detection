import numpy as np
import pandas as pd
import keras.utils as image
from tensorflow import keras

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

# Load the model
base_model = keras.applications.ResNet50(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3))

# Freeze the layers of the model
for layer in base_model.layers:
    layer.trainable = False

# Add new fully connected layers for deepfake detection
x = base_model.output
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(1024, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(1, activation='sigmoid')(x)

# Create the final model
model = keras.models.Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
# model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))
model.fit(x_train, y_train, batch_size=32, epochs=10)