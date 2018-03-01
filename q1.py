#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/21 20:12
# @File    : q1.py
# @Author  : NusLuoKe

import scipy.io
import numpy as np

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


x_train_binarization = binarize_feature(x_train)
x_test_binarization = binarize_feature(x_test)
print(x_train_binarization)

alpha = np.arange(0, 100.5, 0.5)
print(alpha)