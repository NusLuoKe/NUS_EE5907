#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/12 17:50
# @File    : svm.py
# @Author  : NusLuoKe

import numpy as np
from keras.datasets import mnist
from numpy import linalg
from sklearn.svm import SVC

# LOAD MNIST DATA
(X_train, y_train), (X_test, y_test) = mnist.load_data()
x_train = X_train.reshape(X_train.shape[0], 28 * 28 * 1)
x_test = X_test.reshape(X_test.shape[0], 28 * 28 * 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# normalize pixel values to [0, 1]
x_train = x_train / 255.
x_test = x_test / 255.

# # Use raw digit images as inputs to linear SVM
# for c in [0.01, 0.1, 1, 10]:
#     print("SVM classification process begin. C=%s" % c)
#     model = SVC(C=c, kernel="linear", probability=True)
#     model.fit(x_train, y_train)
#     train_acc = model.score(x_train, y_train)
#     test_acc = model.score(x_test, y_test)
#     print("Use raw digit images as inputs to linear SVM when C=%s, training accuracy=%.2f%%" % (c, train_acc * 100))
#     print("Use raw digit images as inputs to linear SVM when C=%s, test accuracy=%.2f%%" % (c, test_acc * 100))

########################################################################
########################################################################
# Use the data vectors after PCA pre-processing as inputs to linear SVM
# Get covariance matrix
mean_train = np.sum(x_train, axis=0) / 60000
mean_test = np.sum(x_test, axis=0) / 10000
zero_mean_train = x_train - mean_train
zero_mean_test = x_test - mean_test
cov_train = np.matmul(zero_mean_train.transpose(), zero_mean_train) / 60000

# Get eigenvalue and eigenvector of the covariance matrix
eig_val_train, eig_vect_train = linalg.eig(np.mat(cov_train))

# Sort the eigenvalue from largest to smallest
sorted_eig_val_train = np.argsort(eig_val_train)[::-1]

# # @@@@@@@@@@@@@@@@@@@@ N=40 start@@@@@@@@@@@@@@@@@@@@@@@
n40_eig_val_train_index = sorted_eig_val_train[0:40]  # take the index of the top n values
# Get the desired eigen vectors and low dimensional data
n40_eig_vect_train = eig_vect_train[:, n40_eig_val_train_index]
x40_train = np.matmul(zero_mean_train, n40_eig_vect_train)
x40_test = np.matmul(zero_mean_test, n40_eig_vect_train)
# for c in [0.01, 0.1, 1, 10]:
#     print("SVM classification process begin.Reduce the dimensionality from 784 to 40, C=%s" % c)
#     model = SVC(C=c, kernel="linear", probability=True)
#     model.fit(x40_train, y_train)
#     train_acc = model.score(x40_train, y_train)
#     test_acc = model.score(x40_test, y_test)
#     print("Reduce the dimensionality from 784 to 40 as inputs to linear SVM when C=%s, training accuracy=%.2f%%" % (
#         c, train_acc * 100))
#     print("Reduce the dimensionality from 784 to 40 as inputs to linear SVM when C=%s, test accuracy=%.2f%%" % (
#         c, test_acc * 100))
# # @@@@@@@@@@@@@@@@@@@@ N=40 end@@@@@@@@@@@@@@@@@@@@@@@@@
#
# # @@@@@@@@@@@@@@@@@@@@ N=80 start@@@@@@@@@@@@@@@@@@@@@@@
# n80_eig_val_train_index = sorted_eig_val_train[0:80]  # take the index of the top n values
# # Get the desired eigen vectors and low dimensional data
# n80_eig_vect_train = eig_vect_train[:, n80_eig_val_train_index]
# x80_train = np.matmul(zero_mean_train, n80_eig_vect_train)
# x80_test = np.matmul(zero_mean_test, n80_eig_vect_train)
# for c in [0.01, 0.1, 1, 10]:
#     print("SVM classification process begin.Reduce the dimensionality from 784 to 80, C=%s" % c)
#     model = SVC(C=c, kernel="linear", probability=True)
#     model.fit(x80_train, y_train)
#     train_acc = model.score(x80_train, y_train)
#     test_acc = model.score(x80_test, y_test)
#     print("Reduce the dimensionality from 784 to 80 as inputs to linear SVM when C=%s, training accuracy=%.2f%%" % (
#         c, train_acc * 100))
#     print("Reduce the dimensionality from 784 to 80 as inputs to linear SVM when C=%s, test accuracy=%.2f%%" % (
#         c, test_acc * 100))
# # @@@@@@@@@@@@@@@@@@@@ N=80 end@@@@@@@@@@@@@@@@@@@@@@@@@

# @@@@@@@@@@@@@@@@@@@@ N=200 start@@@@@@@@@@@@@@@@@@@@@@@
n200_eig_val_train_index = sorted_eig_val_train[0:200]  # take the index of the top n values
# Get the desired eigen vectors and low dimensional data
n200_eig_vect_train = eig_vect_train[:, n200_eig_val_train_index]
x200_train = np.matmul(zero_mean_train, n200_eig_vect_train)
x200_test = np.matmul(zero_mean_test, n200_eig_vect_train)
for c in [0.01, 0.1, 1, 10]:
    print("SVM classification process begin.Reduce the dimensionality from 784 to 200, C=%s" % c)
    model = SVC(C=c, kernel="linear", probability=True)
    model.fit(x200_train, y_train)
    train_acc = model.score(x200_train, y_train)
    test_acc = model.score(x200_test, y_test)
    print("Reduce the dimensionality from 784 to 200 as inputs to linear SVM when C=%s, training accuracy=%.2f%%" % (
        c, train_acc * 100))
    print("Reduce the dimensionality from 784 to 200 as inputs to linear SVM when C=%s, test accuracy=%.2f%%" % (
        c, test_acc * 100))
# @@@@@@@@@@@@@@@@@@@@ N=200 end@@@@@@@@@@@@@@@@@@@@@@@@@


# kernel SVM, Reduce the dimensionality from 784 to 40 as inputs to kernel SVM.
# Tune the parameters C and gamma and observer the performance of the classifier
# @@@@@@@@@@@@@@@@@@@@ N=40 start@@@@@@@@@@@@@@@@@@@@@@@
kernel_svm_train = x40_train
kernel_svm_test = x40_test
for c in [0.01, 0.1, 1, 10]:
    for gma in [0.1, 1, 10, 100]:
        print("SVM classification process begin.Reduce the dimensionality from 784 to 40, C=%s,gamma=%s" % (c, gma))
        model = SVC(C=c, gamma=gma, kernel="rbf", probability=True)
        model.fit(x40_train, y_train)
        train_acc = model.score(x40_train, y_train)
        test_acc = model.score(x40_test, y_test)
        print("Dimensionality 40: C=%s, gamma=%s, training accuracy=%.2f%%" % (
            c, gma, train_acc * 100))
        print("Dimensionality 40: C=%s, gamma=%s, test accuracy=%.2f%%" % (
            c, gma, test_acc * 100))
# @@@@@@@@@@@@@@@@@@@@ N=40 end@@@@@@@@@@@@@@@@@@@@@@@@@
