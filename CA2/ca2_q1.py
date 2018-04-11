#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/11 10:07
# @File    : ca2_q1.py
# @Author  : NusLuoKe


from keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from keras.utils import to_categorical
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from numpy import linalg
import matplotlib.pyplot as plt

###########################################################
#########LOAD MNIST DATA: START############################
###########################################################
(X_train, y_train), (X_test, y_test) = mnist.load_data()
x_train = X_train.reshape(X_train.shape[0], 28 * 28 * 1)
x_test = X_test.reshape(X_test.shape[0], 28 * 28 * 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# normalize pixel values to [0, 1]
x_train = x_train / 255.
x_test = x_test / 255.
# convert class vectors to binary class matrices.
# y_train = to_categorical(y_train, num_classes=10)
# y_test = to_categorical(y_test, num_classes=10)
###########################################################
#########LOAD MNIST DATA: END##############################
###########################################################

# Get covariance matrix
mean_train = np.sum(x_train, axis=0) / 60000
mean_test = np.sum(x_test, axis=0) / 10000
zero_mean_train = x_train - mean_train
zero_mean_test = x_test - mean_test
cov_train = np.matmul(zero_mean_train.transpose(), zero_mean_train) / 60000

# Get eigenvalue and eigenvector of the covariance matrix
eig_val_train, eig_vect_train = linalg.eig(np.mat(cov_train))
eig_val_train_sorted = sorted(eig_val_train, reverse=True)

for n in range(784):
    sum_n_eig_val = sum(eig_val_train_sorted[:n])
    sum_all_eig_val = sum(eig_val_train_sorted)
    rate = sum_n_eig_val / sum_all_eig_val
    if rate > 0.95:
        print(n)
        break

# Sort the eigenvalue from largest to smallest
sorted_eig_val_train = np.argsort(eig_val_train)[::-1]

# @@@@@@@@@@@@@@@@@@@@ N=2 start@@@@@@@@@@@@@@@@@@@@@@@
n2_eig_val_train_index = sorted_eig_val_train[0:2]  # take the index of the top n values
# Get the desired eigen vectors and low dimensional data
n2_eig_vect_train = eig_vect_train[:, n2_eig_val_train_index]
n2_low_dim_data = np.matmul(zero_mean_train, n2_eig_vect_train)
# @@@@@@@@@@@@@@@@@@@@ N=2 end@@@@@@@@@@@@@@@@@@@@@@@@@

# @@@@@@@@@@@@@@@@@@@@ N=2 start@@@@@@@@@@@@@@@@@@@@@@@
n3_eig_val_train_index = sorted_eig_val_train[0:3]  # take the index of the top n values
# Get the desired eigen vectors and low dimensional data
n3_eig_vect_train = eig_vect_train[:, n3_eig_val_train_index]
n3_low_dim_data = np.matmul(zero_mean_train, n3_eig_vect_train)
# @@@@@@@@@@@@@@@@@@@@ N=2 end@@@@@@@@@@@@@@@@@@@@@@@@@


# @@@@@@@@@@@@@@@@@@@@ N=40 start@@@@@@@@@@@@@@@@@@@@@@@
n40_eig_val_train_index = sorted_eig_val_train[0:40]  # take the index of the top n values
# Get the desired eigen vectors and low dimensional data
n40_eig_vect_train = eig_vect_train[:, n40_eig_val_train_index]
x40_train = np.matmul(zero_mean_train, n40_eig_vect_train)
x40_test = np.matmul(zero_mean_test, n40_eig_vect_train)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(x40_train, y_train)

# evaluate the model and update the accuracies list
score = model.score(x40_test, y_test)
print(score * 100)
print("reduce the dimensionality of raw data from 784 to 40, accuracy=%.2f%%" % (score * 100))

# x40_train_binarization = 1 * (x40_train > 0)
# x40_test_binarization = 1 * (x40_test > 0)  # shape is 10000*40
# pred_labels = []
# for test_img_id in range(10000):
#     if test_img_id % 50 == 0:
#         print("Already predicted %d images in test set, There are %d more pictures to calculate" % (
#             test_img_id, 10000 - test_img_id))
#
#     img_test = x40_test_binarization[test_img_id]
#     dist = []
#     for train_img_id in range(60000):
#         img_train = x40_train_binarization[train_img_id]
#
#         aa = (img_test - img_train)
#         a = np.power(aa, 2)
#         b = np.sum(a)
#         dist_a_to_b = np.power(b, 0.5)
#         dist.append(dist_a_to_b)  # find the distance of img in test set to img in training set
#
#     dist_small_to_big = np.argsort(dist)
#     dist_smallest_index = dist_small_to_big[0]
#     pred_label = y_train[dist_smallest_index]
#     pred_labels.append(pred_label)
#
# print(pred_labels)
#
# num_true = sum((pred_labels == y_test) * 1)
# acc = num_true / 10000
# print(acc)
# @@@@@@@@@@@@@@@@@@@@ N=40 end@@@@@@@@@@@@@@@@@@@@@@@@@


# @@@@@@@@@@@@@@@@@@@@ N=80 start@@@@@@@@@@@@@@@@@@@@@@@
n80_eig_val_train_index = sorted_eig_val_train[0:80]  # take the index of the top n values
# Get the desired eigen vectors and low dimensional data
n80_eig_vect_train = eig_vect_train[:, n80_eig_val_train_index]
x80_train = np.matmul(zero_mean_train, n80_eig_vect_train)
x80_test = np.matmul(zero_mean_test, n80_eig_vect_train)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(x80_train, y_train)

# evaluate the model and update the accuracies list
score = model.score(x80_test, y_test)
print("reduce the dimensionality of raw data from 784 to 80, accuracy=%.2f%%" % (score * 100))
# @@@@@@@@@@@@@@@@@@@@ N=80 end@@@@@@@@@@@@@@@@@@@@@@@@@


# @@@@@@@@@@@@@@@@@@@@ N=200 start@@@@@@@@@@@@@@@@@@@@@@@
n200_eig_val_train_index = sorted_eig_val_train[0:200]  # take the index of the top n values
# Get the desired eigen vectors and low dimensional data
n200_eig_vect_train = eig_vect_train[:, n200_eig_val_train_index]
x200_train = np.matmul(zero_mean_train, n200_eig_vect_train)
x200_test = np.matmul(zero_mean_test, n200_eig_vect_train)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(x200_train, y_train)

# evaluate the model and update the accuracies list
score = model.score(x200_test, y_test)
print("reduce the dimensionality of raw data from 784 to 200, accuracy=%.2f%%" % (score * 100))
# @@@@@@@@@@@@@@@@@@@@ N=200 end@@@@@@@@@@@@@@@@@@@@@@@@@
