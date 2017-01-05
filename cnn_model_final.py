from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np

nb_epoch = 20

# input image dimensions
img_rows, img_cols = 32, 32
img_channels = 3


#the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    zca_whitening=False, #change it to true if you need to turn on ZCA whitening. Beware, this will take a lot of time to train since 60000 images have to be whitened.
    height_shift_range=None,
    horizontal_flip=True,
    vertical_flip=False)

curr_batch = 0
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=2048):
    X_train = np.vstack((X_train, X_batch))
    y_train = np.vstack((y_train, y_batch))
    curr_batch += 1

X_train = X_train.astype('float32')
#print(X_train.shape)
X_test = X_test.astype('float32')
#print(X_test.shape)
X_train = X_train / 255
X_test = X_test / 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#print(X_train.shape[1:])
model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print(model.summary())

model.fit(X_train, Y_train,
          batch_size=32,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test))