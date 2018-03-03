#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/3 14:18
# @File    : q3.py
# @Author  : NusLuoKe

from numpy import exp
from math import pi as PI

import numpy as np
import scipy.io
from numpy import log

# load the given spam data
spam_data_path = 'T:/EE5907R/spamData.mat'
spam_data = scipy.io.loadmat(spam_data_path)

# print to see some basic information
print(spam_data.keys())

print("length of training set is:", len(spam_data['Xtrain']))
print("length of test set is:", len(spam_data['Xtest']))

print("shape of x_train: ", spam_data['Xtrain'].shape)
print("shape of y_train:", spam_data['ytrain'].shape)
print("shape of x_test:", spam_data['Xtest'].shape)
print("shape of y_test:", spam_data['ytest'].shape)

# load data
x_train = spam_data['Xtrain']
x_test = spam_data['Xtest']
y_test = spam_data['ytest']
y_train = spam_data['ytrain']


# binarize features
def binarize_feature(mail_array):
    return 1 * (mail_array > 0)


# z-normalise features
def z_normalization(mail_array):
    colum_mean = np.mean(mail_array, axis=0)
    colum_std = np.std(mail_array, axis=0)
    return (mail_array - colum_mean) / colum_std


# log-normalise features
def log_transform(mail_array):
    log_mail_array = log(mail_array + 0.1)
    return log_mail_array


bias_for_feature_train = np.ones((len(x_train), 1))
bias_for_feature_test = np.ones((len(x_test), 1))

x_train_binarization_ = binarize_feature(x_train)
x_test_binarization_ = binarize_feature(x_test)
x_train_binarization = np.hstack((bias_for_feature_train, x_train_binarization_))
x_test_binarization = np.hstack((bias_for_feature_test, x_test_binarization_))

x_train_znorm_ = z_normalization(x_train)
x_test_znorm_ = z_normalization(x_test)
x_train_znorm = np.hstack((bias_for_feature_train, x_train_znorm_))
x_test_znorm = np.hstack((bias_for_feature_test, x_test_znorm_))

x_train_logtrans_ = log_transform(x_train)
x_test_logtrans_ = log_transform(x_test)
x_train_logtrans = np.hstack((bias_for_feature_train, x_train_logtrans_))
x_test_logtrans = np.hstack((bias_for_feature_test, x_test_logtrans_))


def sigmoid(x):
    sigm = 1 / (1 + exp(-x))
    return sigm


lamda_1 = np.arange(1, 10, 1)
lamda_2 = np.arange(10, 105, 5)
lamda_value = np.hstack((lamda_1, lamda_2))
#############################################################
# binarization
for lamda_ in lamda_value:
    omega_with_bias = np.zeros((1, len(x_train[0]) + 1))
    omega_with_bias_T = omega_with_bias.T
    x = np.dot(x_train_binarization, omega_with_bias_T)
    mu = sigmoid(x)
    g = np.dot(x_train_binarization.T, (mu - y_train))

    a = np.zeros((1, 1))
    b = np.ones((1, 57))
    c = np.hstack((a, b))
    lamda = np.diag(c[0]) * lamda_
    g_reg = g + lamda * omega_with_bias.T
    s = np.dot(mu, (1 - mu).T)
    h = np.dot(np.dot(x_train_binarization.T, s), x_train_binarization)


####################################################################
