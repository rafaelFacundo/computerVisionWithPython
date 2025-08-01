#source venv/bin/activate
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.python import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential



dataSetDirectory = "./labeledFramesByFilter/cannyBlur"


image_height = 320
image_width = 320
batch_size = 32 

#loaging the images for the training data set
train_dataSet = tf.keras.utils.image_dataset_from_directory(
    dataSetDirectory,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size,
    color_mode="grayscale"
)

test_dataSet = tf.keras.utils.image_dataset_from_directory(
    dataSetDirectory,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size,
    color_mode="grayscale"
)

classNames = train_dataSet.class_names;
print(f"CLASS NAMES: {classNames}")


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_dataSet.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = test_dataSet.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(classNames)

model = Sequential([
  layers.Rescaling(1./255,input_shape=(image_height, image_width, 1)),  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()




