import os
import pandas as pd
import numpy as np
import cv2
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
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB
                    image = cv2.resize(image, target_size)  # Resize image
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
model.save('my_model.h5')

# Function to preprocess a single image
def preprocess_single_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB
    image = cv2.resize(image, target_size)  # Resize image
    image = image.astype('float32') / 255.0  # Normalize pixel values
    return image

# # Load a single image for prediction
# test_image_path = r'C:\Users\asus\Desktop\ret\uploads\4c570172778b.png'  # Change this to the path of your test image
# test_image = preprocess_single_image(test_image_path)

# # Make predictions on the single image
# predictions = model.predict(np.expand_dims(test_image, axis=0))

# # Get the predicted class label
# predicted_class_index = np.argmax(predictions[0])
# class_labels = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']  # Define class labels
# predicted_class_label = class_labels[predicted_class_index]

# # Print the predicted class label
# print('Predicted class label:', predicted_class_label)

val_loss, val_accuracy = model.evaluate(val_images, val_labels)
print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Make predictions on the validation set
val_predictions = model.predict(val_images)
val_predicted_labels = np.argmax(val_predictions, axis=1)

# Calculate additional evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix
print('Classification Report:')
print(classification_report(val_labels, val_predicted_labels))

print('Confusion Matrix:')
print(confusion_matrix(val_labels, val_predicted_labels))
