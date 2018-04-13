#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/11 10:07
# @File    : ca2_q1.py
# @Author  : NusLuoKe

'''
The size of raw MNIST image is 28  28 pixels, resulting in a 784 dimensional vector for
each image. Apply PCA to reduce the dimensionality of vectorized images from 784 to 2
and 3 respectively. Visualize the projected data vector in 2d and 3d plots. Also visualize the
eigenvectors of sample covariance matrix used in PCA.
Apply PCA to reduce the dimensionality of raw data from 784 to 40, 80 and 200 respectively.
Classifying the test images using the rule of nearest neighbor. Report the classification
accuracy.
Denoted the reduced dimension as d with d  784. Investigate what value of d preserves
over 95% of the total energy after dimensionality reduction. Apply PCA to reduce the data
dimension to d and report classification results based on nearest neighbor. Can you devise
other criteria for automatically determining the value of d?
'''

import numpy as np
from keras.datasets import mnist
from numpy import linalg
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D

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

# Sort the eigenvalue from largest to smallest
sorted_eig_val_train = np.argsort(eig_val_train)[::-1]

# @@@@@@@@@@@@@@@@@@@@ N=2 start@@@@@@@@@@@@@@@@@@@@@@@
n2_eig_val_train_index = sorted_eig_val_train[0:2]  # take the index of the top n values
# Get the desired eigen vectors and low dimensional data
n2_eig_vect_train = eig_vect_train[:, n2_eig_val_train_index]
n2_low_dim_data = np.matmul(zero_mean_train, n2_eig_vect_train)
# @@@@@@@@@@@@@@@@@@@@ N=2 end@@@@@@@@@@@@@@@@@@@@@@@@@

# @@@@@@@@@@@@@@@@@@@@ N=3 start@@@@@@@@@@@@@@@@@@@@@@@
n3_eig_val_train_index = sorted_eig_val_train[0:3]  # take the index of the top n values
# Get the desired eigen vectors and low dimensional data
n3_eig_vect_train = eig_vect_train[:, n3_eig_val_train_index]
n3_low_dim_data = np.matmul(zero_mean_train, n3_eig_vect_train)
# @@@@@@@@@@@@@@@@@@@@ N=3 end@@@@@@@@@@@@@@@@@@@@@@@@@

eigenvector1 = np.reshape(n3_eig_vect_train[:, 0], [28, 28])
eigenvector2 = np.reshape(n3_eig_vect_train[:, 1], [28, 28])
eigenvector3 = np.reshape(n3_eig_vect_train[:, 2], [28, 28])
plt.figure()
plt.subplot(1, 2, 1)
plt.title("eigenvector1")
plt.imshow(eigenvector1, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("eigenvector2")
plt.imshow(eigenvector2, cmap='gray')
plt.suptitle('visualize the eigenvectors in 2d')

plt.figure()
plt.subplot(1, 3, 1)
plt.title("eigenvector1")
plt.imshow(eigenvector1, cmap='gray')
plt.subplot(1, 3, 2)
plt.title("eigenvector2")
plt.imshow(eigenvector2, cmap='gray')
plt.subplot(1, 3, 3)
plt.title("eigenvector3")
plt.imshow(eigenvector3, cmap='gray')
plt.suptitle('visualize the eigenvectors in 3d')
plt.show()

##############################visulize the projection#########################################
img_0_train_2d, img_1_train_2d, img_2_train_2d, img_3_train_2d = [], [], [], []
img_4_train_2d, img_5_train_2d, img_6_train_2d = [], [], []
img_7_train_2d, img_8_train_2d, img_9_train_2d = [], [], []

img_0_train_3d, img_1_train_3d, img_2_train_3d, img_3_train_3d = [], [], [], []
img_4_train_3d, img_5_train_3d, img_6_train_3d = [], [], []
img_7_train_3d, img_8_train_3d, img_9_train_3d = [], [], []
for i in range(60000):
    if y_train[i] == 0:
        img_0_train_2d.append(n2_low_dim_data[i])
        img_0_train_3d.append(n3_low_dim_data[i])
    elif y_train[i] == 1:
        img_1_train_2d.append(n2_low_dim_data[i])
        img_1_train_3d.append(n3_low_dim_data[i])
    elif y_train[i] == 2:
        img_2_train_2d.append(n2_low_dim_data[i])
        img_2_train_3d.append(n3_low_dim_data[i])
    elif y_train[i] == 3:
        img_3_train_2d.append(n2_low_dim_data[i])
        img_3_train_3d.append(n3_low_dim_data[i])
    elif y_train[i] == 4:
        img_4_train_2d.append(n2_low_dim_data[i])
        img_4_train_3d.append(n3_low_dim_data[i])
    elif y_train[i] == 5:
        img_5_train_2d.append(n2_low_dim_data[i])
        img_5_train_3d.append(n3_low_dim_data[i])
    elif y_train[i] == 6:
        img_6_train_2d.append(n2_low_dim_data[i])
        img_6_train_3d.append(n3_low_dim_data[i])
    elif y_train[i] == 7:
        img_7_train_2d.append(n2_low_dim_data[i])
        img_7_train_3d.append(n3_low_dim_data[i])
    elif y_train[i] == 8:
        img_8_train_2d.append(n2_low_dim_data[i])
        img_8_train_3d.append(n3_low_dim_data[i])
    else:
        img_9_train_2d.append(n2_low_dim_data[i])
        img_9_train_3d.append(n3_low_dim_data[i])

img_0_train_2d_ = np.array(img_0_train_2d)
img_1_train_2d_ = np.array(img_1_train_2d)
img_2_train_2d_ = np.array(img_2_train_2d)
img_3_train_2d_ = np.array(img_3_train_2d)
img_4_train_2d_ = np.array(img_0_train_2d)
img_5_train_2d_ = np.array(img_5_train_2d)
img_6_train_2d_ = np.array(img_6_train_2d)
img_7_train_2d_ = np.array(img_7_train_2d)
img_8_train_2d_ = np.array(img_8_train_2d)
img_9_train_2d_ = np.array(img_9_train_2d)

img_0_train_2d = img_0_train_2d_.reshape(img_0_train_2d_.shape[0], img_0_train_2d_.shape[2])
img_1_train_2d = img_1_train_2d_.reshape(img_1_train_2d_.shape[0], img_1_train_2d_.shape[2])
img_2_train_2d = img_2_train_2d_.reshape(img_2_train_2d_.shape[0], img_2_train_2d_.shape[2])
img_3_train_2d = img_3_train_2d_.reshape(img_3_train_2d_.shape[0], img_3_train_2d_.shape[2])
img_4_train_2d = img_4_train_2d_.reshape(img_4_train_2d_.shape[0], img_4_train_2d_.shape[2])
img_5_train_2d = img_5_train_2d_.reshape(img_5_train_2d_.shape[0], img_5_train_2d_.shape[2])
img_6_train_2d = img_6_train_2d_.reshape(img_6_train_2d_.shape[0], img_6_train_2d_.shape[2])
img_7_train_2d = img_7_train_2d_.reshape(img_7_train_2d_.shape[0], img_7_train_2d_.shape[2])
img_8_train_2d = img_8_train_2d_.reshape(img_8_train_2d_.shape[0], img_8_train_2d_.shape[2])
img_9_train_2d = img_9_train_2d_.reshape(img_9_train_2d_.shape[0], img_9_train_2d_.shape[2])

img_0_train_3d_ = np.array(img_0_train_3d)
img_1_train_3d_ = np.array(img_1_train_3d)
img_2_train_3d_ = np.array(img_2_train_3d)
img_3_train_3d_ = np.array(img_3_train_3d)
img_4_train_3d_ = np.array(img_4_train_3d)
img_5_train_3d_ = np.array(img_5_train_3d)
img_6_train_3d_ = np.array(img_6_train_3d)
img_7_train_3d_ = np.array(img_7_train_3d)
img_8_train_3d_ = np.array(img_8_train_3d)
img_9_train_3d_ = np.array(img_9_train_3d)

img_0_train_3d = img_0_train_3d_.reshape(img_0_train_3d_.shape[0], img_0_train_3d_.shape[2])
img_1_train_3d = img_1_train_3d_.reshape(img_1_train_3d_.shape[0], img_1_train_3d_.shape[2])
img_2_train_3d = img_2_train_3d_.reshape(img_2_train_3d_.shape[0], img_2_train_3d_.shape[2])
img_3_train_3d = img_3_train_3d_.reshape(img_3_train_3d_.shape[0], img_3_train_3d_.shape[2])
img_4_train_3d = img_4_train_3d_.reshape(img_4_train_3d_.shape[0], img_4_train_3d_.shape[2])
img_5_train_3d = img_5_train_3d_.reshape(img_5_train_3d_.shape[0], img_5_train_3d_.shape[2])
img_6_train_3d = img_6_train_3d_.reshape(img_6_train_3d_.shape[0], img_6_train_3d_.shape[2])
img_7_train_3d = img_7_train_3d_.reshape(img_7_train_3d_.shape[0], img_7_train_3d_.shape[2])
img_8_train_3d = img_8_train_3d_.reshape(img_8_train_3d_.shape[0], img_8_train_3d_.shape[2])
img_9_train_3d = img_9_train_3d_.reshape(img_9_train_3d_.shape[0], img_9_train_3d_.shape[2])

# plot figure
plt.figure()
plt.title("the first 3000 projected training data vector in 2d")
a0 = plt.scatter(img_0_train_2d[0:3000, 0].real, img_0_train_2d[0:3000, 1].real, c='r', marker='.')
a1 = plt.scatter(img_1_train_2d[0:3000, 0].real, img_1_train_2d[0:3000, 1].real, c='g', marker='x')
a2 = plt.scatter(img_2_train_2d[0:3000, 0].real, img_2_train_2d[0:3000, 1].real, c='b', marker='+')
a3 = plt.scatter(img_3_train_2d[0:3000, 0].real, img_3_train_2d[0:3000, 1].real, c='c', marker='.')
a4 = plt.scatter(img_4_train_2d[0:3000, 0].real, img_4_train_2d[0:3000, 1].real, c='m', marker='x')
a5 = plt.scatter(img_5_train_2d[0:3000, 0].real, img_5_train_2d[0:3000, 1].real, c='y', marker='+')
a6 = plt.scatter(img_6_train_2d[0:3000, 0].real, img_6_train_2d[0:3000, 1].real, c='k', marker='.')
a7 = plt.scatter(img_7_train_2d[0:3000, 0].real, img_7_train_2d[0:3000, 1].real, c='orange', marker='x')
a8 = plt.scatter(img_8_train_2d[0:3000, 0].real, img_8_train_2d[0:3000, 1].real, c='indigo', marker='+')
a9 = plt.scatter(img_9_train_2d[0:3000, 0].real, img_9_train_2d[0:3000, 1].real, c='peru', marker='.')
plt.legend([a0, a1, a2, a3, a4, a5, a6, a7, a8, a9], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
           loc='best')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title("the first 3000 projected training data vector in 3d")
a0 = ax.scatter(img_0_train_3d[0:3000, 0].real, img_0_train_3d[0:3000, 1].real, img_0_train_3d[0:3000, 2].real, c='r',
                marker='.')
a1 = ax.scatter(img_1_train_3d[0:3000, 0].real, img_1_train_3d[0:3000, 1].real, img_1_train_3d[0:3000, 2].real, c='g',
                marker='+')
a2 = ax.scatter(img_2_train_3d[0:3000, 0].real, img_2_train_3d[0:3000, 1].real, img_2_train_3d[0:3000, 2].real, c='b',
                marker='x')
a3 = ax.scatter(img_3_train_3d[0:3000, 0].real, img_3_train_3d[0:3000, 1].real, img_3_train_3d[0:3000, 2].real, c='c',
                marker='.')
a4 = ax.scatter(img_4_train_3d[0:3000, 0].real, img_4_train_3d[0:3000, 1].real, img_4_train_3d[0:3000, 2].real, c='m',
                marker='+')
a5 = ax.scatter(img_5_train_3d[0:3000, 0].real, img_5_train_3d[0:3000, 1].real, img_5_train_3d[0:3000, 2].real, c='y',
                marker='x')
a6 = ax.scatter(img_6_train_3d[0:3000, 0].real, img_6_train_3d[0:3000, 1].real, img_6_train_3d[0:3000, 2].real, c='k',
                marker='.')
a7 = ax.scatter(img_7_train_3d[0:3000, 0].real, img_7_train_3d[0:3000, 1].real, img_7_train_3d[0:3000, 2].real,
                c='orange', marker='+')
a8 = ax.scatter(img_8_train_3d[0:3000, 0].real, img_8_train_3d[0:3000, 1].real, img_8_train_3d[0:3000, 2].real,
                c='indigo', marker='x')
a9 = ax.scatter(img_9_train_3d[0:3000, 0].real, img_9_train_3d[0:3000, 1].real, img_9_train_3d[0:3000, 2].real,
                c='peru', marker='.')
plt.legend([a0, a1, a2, a3, a4, a5, a6, a7, a8, a9], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
           loc='center left')
plt.show()

######################################################################################################
######################################################################################################

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
print("reduce the dimensionality of raw data from 784 to 40, accuracy=%.2f%%" % (score * 100))

# # implement a KNN classifier use binary normalization. TOO SLOW
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


# @@@@@Find reduced dimension d preserving over 95% of total energy@@@@@@@@@
eig_val_train_sorted = sorted(eig_val_train, reverse=True)

for nn in range(784):
    sum_n_eig_val = sum(eig_val_train_sorted[:nn])
    sum_all_eig_val = sum(eig_val_train_sorted)
    rate = sum_n_eig_val / sum_all_eig_val
    if rate > 0.95:
        print(nn, "is the required num!")
        break

nn_eig_val_train_index = sorted_eig_val_train[0:nn]  # take the index of the top n values
# Get the desired eigen vectors and low dimensional data
nn_eig_vect_train = eig_vect_train[:, nn_eig_val_train_index]
xnn_train = np.matmul(zero_mean_train, nn_eig_vect_train)
xnn_test = np.matmul(zero_mean_test, nn_eig_vect_train)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(xnn_train, y_train)

# evaluate the model and update the accuracies list
score = model.score(xnn_test, y_test)
print("reduce the dimensionality of raw data from 784 to %d, accuracy=%.2f%%" % (nn, score * 100))
# @@@@@Find reduced dimension d preserving over 95% of total energy@@@@@@@@@
