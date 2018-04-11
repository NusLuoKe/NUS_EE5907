#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/11 10:07
# @File    : q1.py
# @Author  : NusLuoKe


from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from numpy import linalg

###########################################################
#########LOAD MNIST DATA: START############################
###########################################################
(X_train, y_train), (X_test, y_test) = mnist.load_data()

x_train = X_train.reshape(X_train.shape[0], 28 * 28 * 1)
x_test = X_test.reshape(X_test.shape[0], 28 * 28 * 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.
x_test = x_test / 255.

# convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
# print(x_train.shape)
# print(y_train.shape)
###########################################################
#########LOAD MNIST DATA: END##############################
###########################################################

# Get covariance matrix
mean_train = np.sum(x_train, axis=0) / 60000
zero_mean_train = x_train - mean_train
cov_train = np.matmul(zero_mean_train.transpose(), zero_mean_train) / 60000

# Get eigenvalue and eigenvector of the covariance matrix
eig_val_train, eig_vect_train = linalg.eig(np.mat(cov_train))
print(eig_vect_train.shape)
print(type(eig_vect_train))

# Sort the eigenvalue from largest to smallest
sorted_eig_val_train = np.argsort(eig_val_train)[::-1]
n = 2
n_eig_val_train_index = sorted_eig_val_train[0:n]  # take the index of the top n values

# Get the desired eigen values and eigen vectors
# n_eig_vect_train = eig_vect_train[: n_eig_val_train_index]

# print(n_eig_vect_train.shape)
# print(n_eig_val_train)

# low_dimension_data = np.matmul()