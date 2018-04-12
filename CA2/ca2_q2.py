#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/11 23:13
# @File    : ca2_q2.py
# @Author  : NusLuoKe

'''
Apply LDA to reduce data dimensionality from 784 to 2, 3 and 9. Visualize distribution of the
data with dimensionality of 2 and 3 respectively (similar to PCA). Report the classification accuracy
for data with dimensions of 2, 3 and 9 respectively, based on nearest neighbor classifier.
Test the maximal dimensionality that data can be projected to via LDA. Explain the reasons.
'''
import numpy as np
from keras.datasets import mnist
from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# LOAD MNIST DATA
(X_train, y_train), (X_test, y_test) = mnist.load_data()
x_train = X_train.reshape(X_train.shape[0], 28 * 28 * 1)
x_test = X_test.reshape(X_test.shape[0], 28 * 28 * 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# normalize pixel values to [0, 1]
x_train = x_train / 255.
x_test = x_test / 255.

# Get statistical facts
# Total mean vector
mu_train = np.sum(x_train, axis=0) / 60000
mu_test = np.sum(x_test, axis=0) / 10000

# Get the matrix for each class respectively
img_0_train, img_1_train, img_2_train, img_3_train, img_4_train, img_5_train, img_6_train, img_7_train, img_8_train, img_9_train = [], [], [], [], [], [], [], [], [], []
for i in range(60000):
    if y_train[i] == 0:
        img_0_train.append(x_train[i])
    elif y_train[i] == 1:
        img_1_train.append(x_train[i])
    elif y_train[i] == 2:
        img_2_train.append(x_train[i])
    elif y_train[i] == 3:
        img_3_train.append(x_train[i])
    elif y_train[i] == 4:
        img_4_train.append(x_train[i])
    elif y_train[i] == 5:
        img_5_train.append(x_train[i])
    elif y_train[i] == 6:
        img_6_train.append(x_train[i])
    elif y_train[i] == 7:
        img_7_train.append(x_train[i])
    elif y_train[i] == 8:
        img_8_train.append(x_train[i])
    else:
        img_9_train.append(x_train[i])

img_0_train = np.array(img_0_train)
img_1_train = np.array(img_1_train)
img_2_train = np.array(img_2_train)
img_3_train = np.array(img_3_train)
img_4_train = np.array(img_0_train)
img_5_train = np.array(img_5_train)
img_6_train = np.array(img_6_train)
img_7_train = np.array(img_7_train)
img_8_train = np.array(img_8_train)
img_9_train = np.array(img_9_train)

# Put all the 10 classes in a list
img_trian = []
img_trian.append(img_0_train)
img_trian.append(img_1_train)
img_trian.append(img_2_train)
img_trian.append(img_3_train)
img_trian.append(img_4_train)
img_trian.append(img_5_train)
img_trian.append(img_6_train)
img_trian.append(img_7_train)
img_trian.append(img_8_train)
img_trian.append(img_9_train)

# initialize s_w and s_b
s_w_train = 0
s_b_train = 0

for i in range(10):
    # Class specific mean vector
    mu_i_train = np.mean(img_trian[i], axis=0)

    # Class - specific covariance(scatter) matrix
    zero_mean_train_classed_train = img_trian[i] - mu_i_train
    s_i_train = (1 / img_trian[i].shape[0]) * np.matmul(zero_mean_train_classed_train.transpose(),
                                                        zero_mean_train_classed_train)
    # Within - class scatter
    p_i_train = img_trian[i].shape[0] / 60000
    s_w_i_train = p_i_train * s_i_train
    s_w_train += s_w_i_train

    # Between - class scatter
    mu_temp_train_ = mu_i_train - mu_train
    mu_temp_train = mu_temp_train_.reshape(784, 1)
    s_b_i_train = np.matmul(mu_temp_train, mu_temp_train.transpose()) * p_i_train
    s_b_train += s_b_i_train

#  calculate inv(s_w)*s_b
temp = np.matmul(np.linalg.pinv(s_w_train), s_b_train)

# # Get eigenvalue and eigenvector of the covariance matrix
eig_val_train, eig_vect_train = linalg.eig(np.mat(temp))

# Sort the eigenvalue from largest to smallest
sorted_eig_val_train = np.argsort(eig_val_train)[::-1]

# @@@@@@@@@@@@@@@@@@@@ N=2 start@@@@@@@@@@@@@@@@@@@@@@@
n2_eig_val_train_index = sorted_eig_val_train[0:2]  # take the index of the top n values
# Get the desired eigen vectors and low dimensional data
n2_eig_vect_train = eig_vect_train[:, n2_eig_val_train_index]
x2_train = np.matmul(x_train, n2_eig_vect_train)
x2_test = np.matmul(x_test, n2_eig_vect_train)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(x2_train, y_train)

# evaluate the model and update the accuracies list
score = model.score(x2_test, y_test)
print("reduce the dimensionality of raw data from 784 to 2, accuracy=%.2f%%" % (score * 100))
# @@@@@@@@@@@@@@@@@@@@ N=2 end@@@@@@@@@@@@@@@@@@@@@@@@@

# @@@@@@@@@@@@@@@@@@@@ N=3 start@@@@@@@@@@@@@@@@@@@@@@@
n3_eig_val_train_index = sorted_eig_val_train[0:3]  # take the index of the top n values
# Get the desired eigen vectors and low dimensional data
n3_eig_vect_train = eig_vect_train[:, n3_eig_val_train_index]
x3_train = np.matmul(x_train, n3_eig_vect_train)
x3_test = np.matmul(x_test, n3_eig_vect_train)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(x3_train, y_train)

# evaluate the model and update the accuracies list
score = model.score(x3_test, y_test)
print("reduce the dimensionality of raw data from 784 to 3, accuracy=%.2f%%" % (score * 100))
# @@@@@@@@@@@@@@@@@@@@ N=3 end@@@@@@@@@@@@@@@@@@@@@@@@@


# @@@@@@@@@@@@@@@@@@@@ N=9 start@@@@@@@@@@@@@@@@@@@@@@@
n9_eig_val_train_index = sorted_eig_val_train[0:9]  # take the index of the top n values
# Get the desired eigen vectors and low dimensional data
n9_eig_vect_train = eig_vect_train[:, n9_eig_val_train_index]
x9_train = np.matmul(x_train, n9_eig_vect_train)
x9_test = np.matmul(x_test, n9_eig_vect_train)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(x9_train, y_train)

# evaluate the model and update the accuracies list
score = model.score(x9_test, y_test)
print("reduce the dimensionality of raw data from 784 to 9, accuracy=%.2f%%" % (score * 100))
# @@@@@@@@@@@@@@@@@@@@ N=9 end@@@@@@@@@@@@@@@@@@@@@@@@@

##############################visulize the projection#########################################
img_0_train_2d, img_1_train_2d, img_2_train_2d, img_3_train_2d = [], [], [], []
img_4_train_2d, img_5_train_2d, img_6_train_2d = [], [], []
img_7_train_2d, img_8_train_2d, img_9_train_2d = [], [], []

img_0_train_3d, img_1_train_3d, img_2_train_3d, img_3_train_3d = [], [], [], []
img_4_train_3d, img_5_train_3d, img_6_train_3d = [], [], []
img_7_train_3d, img_8_train_3d, img_9_train_3d = [], [], []
for i in range(60000):
    if y_train[i] == 0:
        img_0_train_2d.append(x2_train[i])
        img_0_train_3d.append(x3_train[i])
    elif y_train[i] == 1:
        img_1_train_2d.append(x2_train[i])
        img_1_train_3d.append(x3_train[i])
    elif y_train[i] == 2:
        img_2_train_2d.append(x2_train[i])
        img_2_train_3d.append(x3_train[i])
    elif y_train[i] == 3:
        img_3_train_2d.append(x2_train[i])
        img_3_train_3d.append(x3_train[i])
    elif y_train[i] == 4:
        img_4_train_2d.append(x2_train[i])
        img_4_train_3d.append(x3_train[i])
    elif y_train[i] == 5:
        img_5_train_2d.append(x2_train[i])
        img_5_train_3d.append(x3_train[i])
    elif y_train[i] == 6:
        img_6_train_2d.append(x2_train[i])
        img_6_train_3d.append(x3_train[i])
    elif y_train[i] == 7:
        img_7_train_2d.append(x2_train[i])
        img_7_train_3d.append(x3_train[i])
    elif y_train[i] == 8:
        img_8_train_2d.append(x2_train[i])
        img_8_train_3d.append(x3_train[i])
    else:
        img_9_train_2d.append(x2_train[i])
        img_9_train_3d.append(x3_train[i])

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
           loc='best')
plt.show()
