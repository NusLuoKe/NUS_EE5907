#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/12 17:40
# @File    : util.py
# @Author  : NusLuoKe

import os

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical


def load_data():
    '''
    Do the pre-process works of MNIST images after load the MNIST data set
    :return: (x_train, y_train), (x_test, y_test)
    '''
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
    return (x_train, y_train), (x_test, y_test)


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



