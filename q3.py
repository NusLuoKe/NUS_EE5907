#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : q3.py
# @Author  : NusLuoKe

import numpy as np
import matplotlib.pyplot as plt
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


def train_w(lamda_, mail_train, mail_label):
    training_step = 0
    max_training_step = 200

    while training_step < max_training_step:
        # get mu
        omega_with_bias = np.zeros((1, len(x_train[0]) + 1))
        omega_with_bias_T = omega_with_bias.T
        x = np.dot(mail_train, omega_with_bias_T)
        mu = sigmoid(x)

        # get g_reg
        g = np.dot(mail_train.T, (mu - mail_label))
        a = np.zeros((1, 1))
        b = np.ones((1, 57))
        c = np.hstack((a, b))
        lamda = np.diag(c[0]) * lamda_
        g_reg = g + np.dot(lamda, omega_with_bias.T)

        # get s
        s_list = []
        for i in mu:
            aaa = i * (1 - i)
            s_list.append(aaa[0])
        s = np.diag(s_list)

        h = np.dot(np.dot(mail_train.T, s), mail_train)
        h_reg = h + lamda * np.eye(58)

        # update omega_with_bias
        HH = np.mat(h_reg)
        try:
            convergence_term = inv(HH) * g_reg
        except np.linalg.linalg.LinAlgError:
            convergence_term = np.linalg.pinv(HH) * g_reg

        omega_with_bias = omega_with_bias - convergence_term.T

        training_step += 1
    return omega_with_bias


def error_rate(mail_test, mail_label, w):
    log_odds = np.dot(mail_test, w.T)
    y_pred = 1 * (log_odds > 0)
    true_num = np.sum(1 * (y_pred == mail_label))
    error_num = len(mail_label) - true_num
    error_rate = error_num / len(mail_label)
    return error_rate


# binarization
for lamda_ in [1, 10, 100]:
    mail_train = x_train_binarization
    mail_label = y_train
    omega_with_bias = train_w(lamda_, mail_train, mail_label)

    mail_test = x_train_binarization
    mail_label = y_train
    w = omega_with_bias
    train_error_rate = error_rate(mail_test, mail_label, w)

    print("error rate for binary-normalization on training set when lamda=%s is:" % lamda_, train_error_rate)

    mail_test = x_test_binarization
    mail_label = y_test
    w = omega_with_bias
    test_error_rate = error_rate(mail_test, mail_label, w)

    print("error rate for binary-normalization on test set when lamda=%s is:" % lamda_, test_error_rate)
    print()

# z-norm
for lamda_ in [1, 10, 100]:
    mail_train = x_train_znorm
    mail_label = y_train
    omega_with_bias = train_w(lamda_, mail_train, mail_label)

    mail_test = x_train_znorm
    mail_label = y_train
    w = omega_with_bias
    train_error_rate = error_rate(mail_test, mail_label, w)

    print("error rate for z-normalization on training set when lamda=%s is:" % lamda_, train_error_rate)

    mail_test = x_test_znorm
    mail_label = y_test
    w = omega_with_bias
    test_error_rate = error_rate(mail_test, mail_label, w)

    print("error rate for z-normalization on test set when lamda=%s is:" % lamda_, test_error_rate)
    print()

# log-trans
for lamda_ in [1, 10, 100]:
    mail_train = x_train_logtrans
    mail_label = y_train
    omega_with_bias = train_w(lamda_, mail_train, mail_label)

    mail_test = x_train_logtrans
    mail_label = y_train
    w = omega_with_bias
    train_error_rate = error_rate(mail_test, mail_label, w)

    print("error rate for log-transformation on training set when lamda=%s is:" % lamda_, train_error_rate)

    mail_test = x_test_logtrans
    mail_label = y_test
    w = omega_with_bias
    test_error_rate = error_rate(mail_test, mail_label, w)

    print("error rate for log-transformation on test set when lamda=%s is:" % lamda_, test_error_rate)
    print()


# plot figure
# log-trans
train_error_ = []
test_error_ = []
for lamda_ in lamda_value:
    mail_train = x_train_logtrans
    mail_label = y_train
    omega_with_bias = train_w(lamda_, mail_train, mail_label)

    mail_test = x_train_logtrans
    mail_label = y_train
    w = omega_with_bias
    train_error_rate = error_rate(mail_test, mail_label, w)
    train_error_.append(train_error_rate)

    mail_test = x_test_logtrans
    mail_label = y_test
    w = omega_with_bias
    test_error_rate = error_rate(mail_test, mail_label, w)
    test_error_.append(test_error_rate)

plt.figure(1)
plt.plot(lamda_value, train_error_, color='r', label='train error rate')
plt.plot(lamda_value, test_error_, color='b', label='test error rate')
plt.title('Error rate - log-trans - logistic regression')
plt.legend(loc=0)
plt.show()

# z-norm
train_error_ = []
test_error_ = []
for lamda_ in lamda_value:
    mail_train = x_train_znorm
    mail_label = y_train
    omega_with_bias = train_w(lamda_, mail_train, mail_label)

    mail_test = x_train_znorm
    mail_label = y_train
    w = omega_with_bias
    train_error_rate = error_rate(mail_test, mail_label, w)
    train_error_.append(train_error_rate)

    mail_test = x_test_znorm
    mail_label = y_test
    w = omega_with_bias
    test_error_rate = error_rate(mail_test, mail_label, w)
    test_error_.append(test_error_rate)

plt.figure(2)
plt.plot(lamda_value, train_error_, color='r', label='train error rate')
plt.plot(lamda_value, test_error_, color='b', label='test error rate')
plt.title('Error rate - z-norm - logistic regression')
plt.legend(loc=0)
plt.show()


# binarization
train_error_ = []
test_error_ = []
for lamda_ in lamda_value:
    mail_train = x_train_binarization
    mail_label = y_train
    omega_with_bias = train_w(lamda_, mail_train, mail_label)

    mail_test = x_train_binarization
    mail_label = y_train
    w = omega_with_bias
    train_error_rate = error_rate(mail_test, mail_label, w)
    train_error_.append(train_error_rate)

    mail_test = x_test_binarization
    mail_label = y_test
    w = omega_with_bias
    test_error_rate = error_rate(mail_test, mail_label, w)
    test_error_.append(test_error_rate)

plt.figure(3)
plt.plot(lamda_value, train_error_, color='r', label='train error rate')
plt.plot(lamda_value, test_error_, color='b', label='test error rate')
plt.title('Error rate - binarization - logistic regression')
plt.legend(loc=0)
plt.show()