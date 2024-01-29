import os

from absl import app
from spoon_detection_model import my_spoon_detection_model

import tensorflow as tf
assert tf.__version__.startswith('2')

IMAGE_SIZE = 96
BATCH_SIZE = 32

BASE_DIR = os.path.join(os.getcwd(), 'spoon-training-images-converted')


def main(argv):
  if len(argv) >= 2:
    model = tf.keras.models.load_model(argv[1])
  else:
    model = my_spoon_detection_model()

  model.summary()

  validation_split = 0.1

  datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rotation_range=10,
      width_shift_range=0.05,
      height_shift_range=0.05,
      zoom_range=.1,
      horizontal_flip=True,
      validation_split=validation_split,
      rescale=1. / 255)
  train_generator = datagen.flow_from_directory(
      BASE_DIR,
      target_size=(IMAGE_SIZE, IMAGE_SIZE),
      batch_size=BATCH_SIZE,
      subset='training',
      color_mode='grayscale')
  val_generator = datagen.flow_from_directory(
      BASE_DIR,
      target_size=(IMAGE_SIZE, IMAGE_SIZE),
      batch_size=BATCH_SIZE,
      subset='validation',
      color_mode='grayscale')
  print(train_generator.class_indices)

  model = train_epochs(model, train_generator, val_generator, 100, 0.0005)

  # Save model HDF5
  if len(argv) >= 3:
    model.save(argv[2])
  else:
    model.save('trained_models/spoon_detection_model.h5')
    model.save('spoon_detection_model.h5')


def train_epochs(model, train_generator, val_generator, epoch_count,
                 learning_rate):
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate),
      loss='categorical_crossentropy',
      metrics=['accuracy'])
  model.fit(
      train_generator,
      steps_per_epoch=len(train_generator),
      epochs=epoch_count,
      validation_data=val_generator,
      validation_steps=len(val_generator),
      batch_size=BATCH_SIZE)
  return model


if __name__ == '__main__':
  app.run(main)