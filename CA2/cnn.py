#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/12 17:37
# @File    : cnn.py
# @Author  : NusLuoKe

import os
import time
import keras
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical


def plot_acc_loss(h, nb_epoch):
    '''
    :param h: history, it is the return value of "fit()", h = model.fit()
    :param nb_epoch: number of epochs
    :return: plot a figure of accuracy and loss of very epoch
    '''
    acc, loss, val_acc, val_loss = h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(range(nb_epoch), acc, label='Train')
    plt.plot(range(nb_epoch), val_acc, label='Test')
    plt.title('Accuracy over ' + str(nb_epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(nb_epoch), loss, label='Train')
    plt.plot(range(nb_epoch), val_loss, label='Test')
    plt.title('Loss over ' + str(nb_epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.show()


# load MNIST data and do pre-works
(X_train, y_train), (X_test, y_test) = mnist.load_data()

x_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
x_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.
x_test = x_test / 255.

# convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
print(x_train.shape)

batch_size = 32
nb_epoch = 1

model = Sequential()
# conv1
model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding="same", input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Activation('relu'))

# con2
model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same'))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Activation('relu'))

# conv3
model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same'))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Activation('relu'))

# hidden
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# output
model.add(Dense(10))
model.add(Activation('softmax'))
adam = keras.optimizers.Adam(lr=1e-4)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
# print the model structure
print(model.summary())

# train the model
start = time.time()
h = model.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size, validation_data=(x_test, y_test),
              shuffle=True, verbose=2, )

# save the model to the following directory
model_dir = 'S:/EE5907_CNN/models'
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
model.save(model_dir + '/MNIST_MODEL_01.h5')

end = time.time()
# print the over all time from training start to the model saved
print('@ Total Time Spent: %.2f seconds' % (end - start))

# plot figures of accuracy and loss of every epoch and a visible test result
plot_acc_loss(h, nb_epoch)

# print loss and accuracy on the whole training set and test set
loss, accuracy = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy = %.2f %%     loss = %f" % (accuracy * 100, loss))
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy = %.2f %%    loss = %f" % (accuracy * 100, loss))
