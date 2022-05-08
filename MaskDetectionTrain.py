import numpy as np
import os
import PIL
import tensorflow as tf

dataset_dir = "traindata"
testdata_dir = "testdata"

batch_size = 1
img_height = 280
img_width = 280

data = tf.keras.utils.image_dataset_from_directory(
  dataset_dir,
  color_mode='grayscale',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs")



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
model.add(tf.keras.layers.Dense(128,activation='relu'))

model.add(tf.keras.layers.Dense(2,activation='softmax'))

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

model.fit(data,epochs=5,batch_size=1,callbacks=[tensorboard])

data = tf.keras.utils.image_dataset_from_directory( #Loads in separate training data to test the model
  testdata_dir,
  color_mode='grayscale',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

model.evaluate(data, batch_size=1)

model.save('model')