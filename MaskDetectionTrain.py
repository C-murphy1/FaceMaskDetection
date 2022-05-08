import numpy as np
import os
import PIL
import tensorflow as tf

dataset_dir = "traindata"

batch_size = 4
img_height = 180
img_width = 180

data = tf.keras.utils.image_dataset_from_directory(
  dataset_dir,
  color_mode='grayscale',
  validation_split=0.4,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


optimizer = tf.keras.optimizers.Adagrad(0.01)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(16,3,padding='same',activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(32,3,padding='same',activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation='relu'))

model.add(tf.keras.layers.Dense(2,activation='softmax'))

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

model.fit(data,epochs=5,batch_size=4)

model.save('model')