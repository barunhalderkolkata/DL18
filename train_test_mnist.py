#!/usr/bin/python
# This code was developed using https://medium.com/@the1ju/simple-logistic-regression-using-keras-249e0cc9a970
# Build the model of a logistic classifier
import os
import gzip
import six.moves.cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD


def build_logistic_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))

    return model

def create_model(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, learning_rate, batch_size, nb_epoch, nb_classes):
	print("Model is building for LR:", learning_rate, ", batch_size:", batch_size)
	model = build_logistic_model(input_dim, nb_classes)

	model.summary()

	# compile the model
	sgd=SGD(lr=learning_rate)
	model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
	history = model.fit(X_train, Y_train,
	                    batch_size=batch_size, nb_epoch=nb_epoch,
	                    verbose=1, validation_data=(X_valid, Y_valid))
	print('History Data:', history.history.keys())
	print('History Data:', history.history['val_acc'])
	score = model.evaluate(X_test, Y_test, verbose=0)

	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	x_axes = np.arange(1, nb_epoch+1)

	# setting a style to use
	plt.style.use('fivethirtyeight')

	# create a figure
	fig = plt.figure()
	 
	# define subplots and their positions in figure
	plt1 = fig.add_subplot(121)
	plt2 = fig.add_subplot(122)

	# plotting the points 
	plt1.plot(x_axes, history.history['val_loss'], linestyle='dashed', linewidth = 3,
	         marker='o', markerfacecolor='yellow', markersize=6)
	# setting x and y axis range
	# plt.ylim(0,1)
	# plt2.xlim(0,16)
	 
	# giving a title to my graph
	plt1.set_title('Val Loss')

	# plotting the points 
	plt2.plot(x_axes, history.history['val_acc'], linestyle='dashed', linewidth = 3,
	         marker='o', markerfacecolor='blue', markersize=6)
	# setting x and y axis range
	# plt.ylim(0,1)
	# plt2.xlim(0,16)
	 
	# giving a title to my graph
	plt2.set_title('Val Accuracy')

	# adjusting space between subplots
	fig.subplots_adjust(hspace=.5,wspace=0.5)

	title= 'Learning Rate:' + str(learning_rate) + ', Batch Size:' + str(batch_size)+"\nVal Acc:" + str(history.history['val_acc'][-1]) + "\nTest Accuracy:" + str(score[1])
	fig_title= 'LR-' + str(learning_rate) + '-BS-' + str(batch_size)
	file_name = 'lr-'+str(learning_rate)+'batch-'+str(batch_size)+'.png'

	fig.canvas.set_window_title(fig_title)
	plt.suptitle(title) 
	# plt.savefig(file_name, bbox_inches='tight')
	# function to show the plot
	plt.show()




learning_rate = 0.1
batch_size = 128
nb_classes = 10
nb_epoch = 15
input_dim = 784

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, input_dim)
X_test = X_test.reshape(10000, input_dim)
val_indices = np.random.permutation(X_train.shape[0])
validation_indices, training_indices = val_indices[:10000], val_indices[10000:]
X_valid = X_train[validation_indices, :]
X_train = X_train[training_indices, :]
y_valid = y_train[validation_indices]
y_train = y_train[training_indices]
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_valid /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# print(y_train.shape, 'label samples shape')
# print(y_test.shape, 'label samples shape')


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_valid = np_utils.to_categorical(y_valid, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
# print(Y_train.shape, 'label samples shape after hot-encode')
# print(Y_test.shape, 'label samples shape after hot-encode')

lrs = [0.001, 0.01, 0.05, 0.1]
batch_sizes = [1024, 128, 32, 1]
for lr in lrs:
	for batch_size in batch_sizes:
		create_model(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, lr, batch_size, nb_epoch, nb_classes)


