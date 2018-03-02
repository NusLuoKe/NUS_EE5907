#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/2 23:18
# @File    : q2.py
# @Author  : NusLuoKe


import scipy.io
import numpy as np
from numpy import log
import matplotlib.pyplot as plt

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


# z-normalise features
def z_norm(mail_array):
    colum_mean = np.mean(mail_array, axis=0)
    colum_stddev = np.sum((mail_array - colum_mean)**2, axis=0) / len(mail_array)
    cd = np.std(mail_array, axis=0)
    return (mail_array - colum_mean) / colum_stddev


# log-normalise features
def log_norm(mail_array):
    log_mail_array = log(mail_array + 0.1)
    return log_mail_array
