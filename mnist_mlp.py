'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.optimizers import SGD
import numpy as np

learning_rate = 0.1
batch_size = 32
num_classes = 10
epochs = 20
input_dim = 784
hidden_layer_activation = 512

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, input_dim )
x_test = x_test.reshape(10000, input_dim)
val_indices = np.random.permutation(x_train.shape[0])
validation_indices, training_indices = val_indices[:10000], val_indices[10000:]
x_valid = x_train[validation_indices, :]
x_train = x_train[training_indices, :]
y_valid = y_train[validation_indices]
y_train = y_train[training_indices]
x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_valid /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, num_classes)
y_valid = np_utils.to_categorical(y_valid, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(hidden_layer_activation, activation='relu', input_shape=(input_dim,)))
model.add(Dense(hidden_layer_activation, activation='relu'))
model.add(Dense(hidden_layer_activation, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

sgd=SGD(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    nb_epoch=epochs,
                    verbose=1,
                    validation_data=(x_valid, y_valid))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
