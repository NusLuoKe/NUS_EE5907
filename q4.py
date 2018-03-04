#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : q4.py
# @Author  : NusLuoKe

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from numpy import log

# load the given spam data
# spam_data_path = 'T:/EE5907R/spamData.mat'
# spam_data = scipy.io.loadmat(spam_data_path)
spam_data = scipy.io.loadmat('spamData.mat')

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


x_train_binarization = binarize_feature(x_train)
x_test_binarization = binarize_feature(x_test)

x_train_znorm = z_normalization(x_train)
x_test_znorm = z_normalization(x_test)

x_train_logtrans = log_transform(x_train)
x_test_logtrans = log_transform(x_test)

k_1 = np.arange(1, 10, 1)
k_2 = np.arange(10, 105, 5)
k_value = np.hstack((k_1, k_2))


##############################################################
##############################################################
# method for z_norm and log_trans
##############################################################
##############################################################

def pred_spam(k, mail, mail_label, train_mail):
    mail_pred = []
    for mail_a_id in range(len(mail)):
        mail_a = mail[mail_a_id]
        dist_mail_single_a = []
        for mail_b_id in range(len(train_mail)):
            mail_b = train_mail[mail_b_id]

            a = ((mail_a - mail_b) ** 2)
            b = np.sum(a)
            dist_a_to_b = b ** 0.5

            dist_mail_single_a.append(dist_a_to_b)  # 找到了a邮件距离别的邮件的距离

        dist_small_to_big = np.argsort(dist_mail_single_a)  # 找到了距离最近的K封邮件的索引

        spam_counter = 0
        mail_index = dist_small_to_big[:k]
        for i in mail_index:
            if mail_label[i] == 1:
                spam_counter += 1

        p_spam = spam_counter / k

        if p_spam <= 0.5:
            y_pred = 0
        elif p_spam > 0.5:
            y_pred = 1
        mail_pred.append(y_pred)

    return mail_pred


def error_rate(mail_pred, mail_label):
    mail_label_ = mail_label.T[0]
    true_num = np.sum((mail_label_ == mail_pred) * 1)
    wrong_num = len(mail_pred) - true_num
    error_rate = wrong_num / len(mail_pred)
    return error_rate


# z_norm
# k = 1
mail_pred = pred_spam(1, x_train_znorm, y_train, x_train_znorm)
print("error rate for z-normalization on training set when k=1 is:", error_rate(mail_pred, y_train))
mail_pred = pred_spam(1, x_test_znorm, y_train, x_train_znorm)
print("error rate for z-normalization on test set when k=1 is:", error_rate(mail_pred, y_test))

# k = 10
mail_pred = pred_spam(10, x_train_znorm, y_train, x_train_znorm)
print("error rate for z-normalization on training set when k=10 is:", error_rate(mail_pred, y_train))
mail_pred = pred_spam(10, x_test_znorm, y_train, x_train_znorm)
print("error rate for z-normalization on test set when k=10 is:", error_rate(mail_pred, y_test))

# k = 100
mail_pred = pred_spam(100, x_train_znorm, y_train, x_train_znorm)
print("error rate for z-normalization on training set when k=100 is:", error_rate(mail_pred, y_train))
mail_pred = pred_spam(100, x_test_znorm, y_train, x_train_znorm)
print("error rate for z-normalization on test set when k=100 is:", error_rate(mail_pred, y_test))

print()
print()

# log_trans
# k = 1
mail_pred = pred_spam(1, x_train_logtrans, y_train, x_train_logtrans)
print("error rate for log-transform on training set when k=1 is:", error_rate(mail_pred, y_train))
mail_pred = pred_spam(1, x_test_logtrans, y_train, x_train_logtrans)
print("error rate for log-transform on test set when k=1 is:", error_rate(mail_pred, y_test))

# k = 10
mail_pred = pred_spam(10, x_train_logtrans, y_train, x_train_logtrans)
print("error rate for log-transform on training set when k=10 is:", error_rate(mail_pred, y_train))
mail_pred = pred_spam(10, x_test_logtrans, y_train, x_train_logtrans)
print("error rate for log-transform on test set when k=10 is:", error_rate(mail_pred, y_test))

# k = 100
mail_pred = pred_spam(100, x_train_logtrans, y_train, x_train_logtrans)
print("error rate for log-transform on training set when k=100 is:", error_rate(mail_pred, y_train))
mail_pred = pred_spam(100, x_test_logtrans, y_train, x_train_logtrans)
print("error rate for log-transform on test set when k=100 is:", error_rate(mail_pred, y_test))
print()
print()


##############################################################
##############################################################
# method for z_norm and log_trans
##############################################################
##############################################################
# method for binarize features
def pred_spam_binarization(k, mail, mail_label, train_mail):
    mail_pred = []
    for mail_a_id in range(len(mail)):
        mail_a = mail[mail_a_id]
        dist_mail_single_a = []
        for mail_b_id in range(len(train_mail)):
            mail_b = train_mail[mail_b_id]

            a = abs(mail_a - mail_b)
            b = np.sum(a)
            dist_a_to_b = b

            dist_mail_single_a.append(dist_a_to_b)

        dist_small_to_big = np.argsort(dist_mail_single_a)

        spam_counter = 0
        mail_index = dist_small_to_big[:k]
        for i in mail_index:
            if mail_label[i] == 1:
                spam_counter += 1

        p_spam = spam_counter / k

        if p_spam <= 0.5:
            y_pred = 0
        elif p_spam > 0.5:
            y_pred = 1
        mail_pred.append(y_pred)

    return mail_pred


# binarization
# k = 1
mail_pred = pred_spam(1, x_train_binarization, y_train, x_train_binarization)
print("error rate for binarization on training set when k=1 is:", error_rate(mail_pred, y_train))
mail_pred = pred_spam(1, x_test_binarization, y_train, x_train_binarization)
print("error rate for binarization on test set when k=1 is:", error_rate(mail_pred, y_test))

# k = 10
mail_pred = pred_spam(10, x_train_binarization, y_train, x_train_binarization)
print("error rate for binarization on training set when k=10 is:", error_rate(mail_pred, y_train))
mail_pred = pred_spam(10, x_test_binarization, y_train, x_train_binarization)
print("error rate for binarization on test set when k=10 is:", error_rate(mail_pred, y_test))

# k = 100
mail_pred = pred_spam(100, x_train_binarization, y_train, x_train_binarization)
print("error rate for binarization on training set when k=100 is:", error_rate(mail_pred, y_train))
mail_pred = pred_spam(100, x_test_binarization, y_train, x_train_binarization)
print("error rate for binarization on test set when k=100 is:", error_rate(mail_pred, y_test))
print()
print()

# z_norm
train_error_ = []
test_error_ = []
for k in k_value:
    mail_train_pred = pred_spam(k, x_train_znorm, y_train, x_train_znorm)
    train_error_rate = error_rate(mail_train_pred, y_train)
    train_error_.append(train_error_rate)

    mail_test_pred = pred_spam(k, x_test_znorm, y_train, x_train_znorm)
    test_error_rate = error_rate(mail_test_pred, y_test)
    test_error_.append(test_error_rate)

plt.figure(1)
plt.plot(k_value, train_error_, color='r', label='train error rate')
plt.plot(k_value, test_error_, color='b', label='test error rate')
plt.title('Error rate - z_norm - KNN')
plt.legend(loc=0)
plt.show()

# log_transform
train_error_ = []
test_error_ = []
for k in k_value:
    mail_train_pred = pred_spam(k, x_train_logtrans, y_train, x_train_logtrans)
    train_error_rate = error_rate(mail_train_pred, y_train)
    train_error_.append(train_error_rate)

    mail_test_pred = pred_spam(k, x_test_logtrans, y_train, x_train_logtrans)
    test_error_rate = error_rate(mail_test_pred, y_test)
    test_error_.append(test_error_rate)

plt.figure(2)
plt.plot(k_value, train_error_, color='r', label='train error rate')
plt.plot(k_value, test_error_, color='b', label='test error rate')
plt.title('Error rate - log_transform - KNN')
plt.legend(loc=0)
plt.show()

# binarization
train_error_ = []
test_error_ = []
for k in k_value:
    mail_train_pred = pred_spam(k, x_train_binarization, y_train, x_train_binarization)
    train_error_rate = error_rate(mail_train_pred, y_train)
    train_error_.append(train_error_rate)

    mail_test_pred = pred_spam(k, x_test_binarization, y_train, x_train_binarization)
    test_error_rate = error_rate(mail_test_pred, y_test)
    test_error_.append(test_error_rate)


plt.figure(3)
plt.plot(k_value, train_error_, color='r', label='train error rate')
plt.plot(k_value, test_error_, color='b', label='test error rate')
plt.title('Error rate - binarization - KNN')
plt.legend(loc=0)
plt.show()
