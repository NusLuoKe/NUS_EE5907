#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/3 14:18
# @File    : q3.py
# @Author  : NusLuoKe

import numpy as np
import scipy.io
from numpy import exp
from numpy import log
from numpy.linalg import inv

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
# for lamda_ in lamda_value:
for lamda_ in [2]:
    # get mu
    omega_with_bias = np.zeros((1, len(x_train[0]) + 1))
    omega_with_bias_T = omega_with_bias.T
    x = np.dot(x_train_binarization, omega_with_bias_T)
    mu = sigmoid(x)
    # print(mu)
    # print(mu.shape)

    print("bbbbbbbbbbaaaaaaaaaa", omega_with_bias)
    print("bbbbbbbbbaaaaaaaaaab", omega_with_bias.shape)

    # get g_reg
    g = np.dot(x_train_binarization.T, (mu - y_train))
    a = np.zeros((1, 1))
    b = np.ones((1, 57))
    c = np.hstack((a, b))
    lamda = np.diag(c[0]) * lamda_
    g_reg = g + np.dot(lamda, omega_with_bias.T)
    # print(g_reg)
    # print(g_reg.shape)

    # get s
    s_list = []
    for i in mu:
        aaa = i*(1-i)
        s_list.append(aaa[0])
    s = np.diag(s_list)
    # print(s)
    # print(s.shape)


    #get h_reg
    h = np.dot(np.dot(x_train_binarization.T, s), x_train_binarization)
    h_reg = h + lamda * np.eye(58)
    # print(h_reg)
    # print(h_reg.shape)

    training_step = 0
    max_training_step = 200
    threashold = np.ones((1, 58)) * 0.0001

    while training_step < max_training_step:
        # get mu
        omega_with_bias = np.zeros((1, len(x_train[0]) + 1))
        omega_with_bias_T = omega_with_bias.T
        x = np.dot(x_train_binarization, omega_with_bias_T)
        mu = sigmoid(x)
        # print(mu)
        # print(mu.shape)

        # get g_reg
        g = np.dot(x_train_binarization.T, (mu - y_train))
        a = np.zeros((1, 1))
        b = np.ones((1, 57))
        c = np.hstack((a, b))
        lamda = np.diag(c[0]) * lamda_
        g_reg = g + np.dot(lamda, omega_with_bias.T)
        # print(g_reg)

        # get s
        s_list = []
        for i in mu:
            aaa = i * (1 - i)
            s_list.append(aaa[0])
        s = np.diag(s_list)
        # print(s)
        # print(s.shape)

        # get h_reg
        h = np.dot(np.dot(x_train_binarization.T, s), x_train_binarization)
        h_reg = h + lamda * np.eye(58)

        # update omega_with_bias
        HH = np.mat(h_reg)
        try:
            convergence_term = inv(HH) * g_reg
        except np.linalg.linalg.LinAlgError:
            convergence_term = np.linalg.pinv(HH) * g_reg

        omega_with_bias = omega_with_bias - convergence_term.T

        training_step += 1

# print("bbbbbbbbbb", omega_with_bias)
print("bbbbbbbbbb", omega_with_bias.shape)

log_odds = np.dot(x_train_binarization, omega_with_bias.T)
print(log_odds.shape)
print(log_odds)
y_pred = 1 * (log_odds > 0)
print(y_pred)
true_num = np.sum(1 * (y_pred == y_train))
error_num = len(y_train) - true_num
error_rate = error_num / len(y_train)
print(error_num)
print(error_rate)