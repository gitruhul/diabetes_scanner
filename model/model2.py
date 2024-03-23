import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from math import ceil

# Function to load and preprocess images
def load_and_preprocess_images(main_dir, csv_file, target_size=(128, 128)):
    # Load the CSV file
    data = pd.read_csv(csv_file)

    images = []
    labels = []

    for subdir in os.listdir(main_dir):
        sub_dir = os.path.join(main_dir, subdir)
        if os.path.isdir(sub_dir):
            for image_file in os.listdir(sub_dir):
                if image_file.endswith('.png'):  # Assuming images are in PNG format
                    image_path = os.path.join(sub_dir, image_file)
                    image = imread(image_path)
                    image = resize(image, target_size)  # Resize images to desired size
                    image = image.astype('float32') / 255.0  # Normalize pixel values

                    images.append(image)

                    # Get class label from CSV file based on image filename
                    image_code = os.path.splitext(image_file)[0]
                    label = data.loc[data['id_code'] == image_code, 'diagnosis'].values[0]
                    labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Directory containing subfolders with images
main_dir = r'C:\Users\asus\Desktop\ret\dataset\gaussian_filtered_images'
# CSV file containing image labels
csv_file = r'C:\Users\asus\Desktop\ret\dataset\train.csv'

# Load and preprocess images with reduced size
images, labels = load_and_preprocess_images(main_dir, csv_file, target_size=(128, 128))

# Encode class labels if necessary
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the dataset into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='softmax')  # Assuming 5 classes
])


# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Learning rate reduction and early stopping callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with data augmentation
history = model.fit(datagen.flow(train_images, train_labels, batch_size=64, shuffle=True),  # Add shuffle=True
                    steps_per_epoch=ceil(len(train_images) / 64),  
                    epochs=5,
                    validation_data=(val_images, val_labels),
                    callbacks=[reduce_lr, early_stop])


# Save the trained model
model.save('model2.h5')

