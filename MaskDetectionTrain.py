import os
import PIL
import tensorflow as tf

traindata_dir = "traindata"
testdata_dir = "testdata"
validationdata_dir = "validationdata"

batch_size = 1
img_height = 280
img_width = 280

traindata = tf.keras.utils.image_dataset_from_directory(
  traindata_dir,
  color_mode='grayscale',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

validationdata = tf.keras.utils.image_dataset_from_directory(
  validationdata_dir,
  color_mode='grayscale',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

testdata = tf.keras.utils.image_dataset_from_directory( #Loads in separate training data to test the model
  testdata_dir,
  color_mode='grayscale',
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

model.fit(traindata,epochs=5,batch_size=1,callbacks=[tensorboard],validation_data=validationdata)

model.evaluate(testdata, batch_size=1)

model.save('model')
