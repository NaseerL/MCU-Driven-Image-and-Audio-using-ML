import tensorflow as tf
import numpy as np
import os

IMAGE_SIZE = 96
BATCH_SIZE = 32


BASE_DIR = os.path.join(os.getcwd(), 'spoon-training-images-converted')

# Load the saved model
model = tf.keras.models.load_model('trained_models/spoon_detection_model.h5')

# Create an ImageDataGenerator for the test dataset
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Load the test dataset from the reduced_dataset directory
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=.1,
    horizontal_flip=True,
    validation_split=0.1,
    rescale=1. / 255)
train_generator = datagen.flow_from_directory(
    BASE_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='training',
    color_mode='grayscale')

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(train_generator, batch_size=BATCH_SIZE)
print('Test accuracy:', accuracy)
