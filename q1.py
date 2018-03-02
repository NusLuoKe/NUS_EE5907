#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/21 20:12
# @File    : q1.py
# @Author  : NusLuoKe

import scipy.io
import numpy as np
from math import log

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


def binarize_feature(mail_array):
    '''
    :param mail_array: mail_array with 57 features
    :return: if feature is 0, keep 0, otherwise set as 1
    '''
    return 1 * (mail_array > 0)


# binarize features
x_train_binarization = binarize_feature(x_train)
x_test_binarization = binarize_feature(x_test)

# get pi_c = P(y=c|T) of training set, here c=1 means spam
num_train_mail = len(y_train)
num_train_spam = int(np.sum(y_train, axis=0))
num_train_not_spam = num_train_mail - num_train_spam
pi_1 = num_train_spam / num_train_mail
pi_0 = 1 - pi_1
pi_c = np.hstack((pi_1, pi_0))
print("pi_c: ", pi_c)

# P(theta_jc) is Beta distribution, P(theta_jc) = Beta(theta_jc|alpha, alpha)
# here alpha is 0, 0.5, 1, ..., 99.5, 100
alpha_value = np.arange(0, 100.5, 0.5)

# for j from 1 to 57, we calculate theta_jc = P(xj=1|y=1,T) = (N_jc + alpha) / (N_c + 2*alpha)
N_1_ = num_train_spam
N_1 = N_1_ * np.ones((57,))
N_0_ = num_train_not_spam
N_0 = N_0_ * np.ones((57,))
N_c = np.vstack((N_1, N_0))

# get the index of spam and not-spam in training set
spam_train_index = []
not_spam_train_index = []
counter = 0
for i in y_train:
    if i == 1:
        spam_train_index.append(counter)
    else:
        not_spam_train_index.append(counter)
    counter += 1

# N_jc
N_j1 = np.sum(x_train[spam_train_index], axis=0)
N_j0 = np.sum(x_train[not_spam_train_index], axis=0)
print("aaaa", N_j1[56])
print("aaaa", N_j0[56])
N_jc = np.vstack((N_j1, N_j0))



# theta_jc.shape = (2, 57)，first row is the prob for each feature when the mail is a spam
# python numpy,alpha can be a scalar
def cal_theta_jc(alpha):
    theta_jc = (N_jc + alpha) / (N_c + 2 * alpha)
    return theta_jc


# classify
#############################################################################
# for alpha in alpha_value:
#     theta_jc = cal_theta_jc(alpha)
#     print(theta_jc.shape)
#     # P(y=c|x,T) -> pi_c * PROD: theta_jc[xj==1]*(1-theta_jc)[xj==0]
#     # P_pred = pi_c *
#
#     break
#############################################################################
error_counter = 0
theta_jc = cal_theta_jc(1)
print(theta_jc)
log_p_pred_spam = np.empty((len(x_train), 1))
log_p_pred_not_spam = np.empty((len(x_train), 1))
# for mail_id in range(len(x_train)):
for mail_id in range(5):
    a_not_spam = 0
    a_spam = 0
    for feature_id in range(len(x_train[mail_id])):
        b_spam = x_train[mail_id][feature_id] * log(theta_jc[0][feature_id]) + (
                1 - x_train[mail_id][feature_id]) * (1 - theta_jc[0][feature_id])
        a_spam = a_spam + b_spam

        b_not_spam = x_train[mail_id][feature_id] * log(theta_jc[1][feature_id]) + (
                1 - x_train[mail_id][feature_id]) * (1 - theta_jc[1][feature_id])
        a_not_spam = a_not_spam + b_not_spam

    log_p_pred_spam[mail_id] = log(pi_c[0]) + a_spam
    log_p_pred_not_spam[mail_id] = log(pi_c[1]) + a_not_spam

    if log_p_pred_spam[mail_id] > log_p_pred_not_spam[mail_id]:
        y_pred = 1
    else:
        y_pred = 0

    if y_pred != y_train[mail_id]:
        error_counter += 1

    print(y_pred)
    print(y_train[mail_id])
    print(error_counter)
    print()
error_rate = error_counter / len(y_train)

print(error_rate)
