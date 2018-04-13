#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : gaussian_naive_bayes.py
# @Author  : NusLuoKe


from math import exp
from math import pi as PI

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


# z-normalise features
def z_normalization(mail_array):
    colum_mean = np.mean(mail_array, axis=0)
    colum_std = np.std(mail_array, axis=0)
    return (mail_array - colum_mean) / colum_std


# log-normalise features
def log_transform(mail_array):
    log_mail_array = log(mail_array + 0.1)
    return log_mail_array


x_train_znorm = z_normalization(x_train)
x_test_znorm = z_normalization(x_test)

x_train_logtrans = log_transform(x_train)
x_test_logtrans = log_transform(x_test)

# get pi_c = P(y=c|T) of training set, here c=1 means spam
num_train_mail = len(y_train)
num_train_spam = int(np.sum(y_train, axis=0))
num_train_not_spam = num_train_mail - num_train_spam
pi_1 = num_train_spam / num_train_mail
pi_0 = 1 - pi_1
pi_c = np.hstack((pi_1, pi_0))
print("pi_c: ", pi_c)
print()

# N_c is the number of samples in class c
N_1 = num_train_spam
N_0 = num_train_not_spam
N_c = np.vstack((N_1, N_0))

# get the index of spam and not-spam in training set
spam_train_index = []
not_spam_train_index = []
mail_index = 0
for i in y_train:
    if i == 1:
        spam_train_index.append(mail_index)
    else:
        not_spam_train_index.append(mail_index)
    mail_index += 1

mu_z_spam = np.mean(x_train_znorm[spam_train_index], axis=0)
mu_z_not_spam = np.mean(x_train_znorm[not_spam_train_index], axis=0)
mu_z = np.vstack((mu_z_spam, mu_z_not_spam))

sigma_z_spam = np.std(x_train_znorm[spam_train_index], axis=0)
sigma_z_not_spam = np.std(x_train_znorm[not_spam_train_index], axis=0)
simg_z = np.vstack((sigma_z_spam, sigma_z_not_spam))

mu_log_spam = np.mean(x_train_logtrans[spam_train_index], axis=0)
mu_log_not_spam = np.mean(x_train_logtrans[not_spam_train_index], axis=0)
mu_log = np.vstack((mu_log_spam, mu_log_not_spam))

sigma_log_spam = np.std(x_train_logtrans[spam_train_index], axis=0)
sigma_log_not_spam = np.std(x_train_logtrans[not_spam_train_index], axis=0)
simg_log = np.vstack((sigma_log_spam, sigma_log_not_spam))


def gaussian_pdf(x, mu, sigma):
    p = (1 / ((2 * PI * (sigma ** 2)) ** 0.5)) * exp((-0.5 * ((x - mu) ** 2)) / (sigma ** 2))
    return p


def classify_error_znorm(emails, emails_label):
    error_counter = 0
    p_pred_spam = np.empty((len(emails), 1))
    p_pred_not_spam = np.empty((len(emails), 1))
    for mail_id in range(len(emails)):
        a_not_spam = 1
        a_spam = 1
        for feature_id in range(len(emails[mail_id])):
            b_spam = gaussian_pdf(emails[mail_id][feature_id], mu_z[0][feature_id], simg_z[0][feature_id])
            a_spam = a_spam * b_spam

            b_not_spam = gaussian_pdf(emails[mail_id][feature_id], mu_z[1][feature_id], simg_z[1][feature_id])
            a_not_spam = a_not_spam * b_not_spam

        p_pred_spam[mail_id] = pi_c[0] * a_spam
        p_pred_not_spam[mail_id] = pi_c[1] * a_not_spam

        if p_pred_spam[mail_id] > p_pred_not_spam[mail_id]:
            y_pred = 1
        else:
            y_pred = 0

        if y_pred != emails_label[mail_id]:
            error_counter += 1

        error_rate = error_counter / len(emails_label)

    return error_rate


def classify_error_logtrans(emails, emails_label):
    error_counter = 0
    p_pred_spam = np.empty((len(emails), 1))
    p_pred_not_spam = np.empty((len(emails), 1))
    for mail_id in range(len(emails)):
        a_not_spam = 1
        a_spam = 1
        for feature_id in range(len(emails[mail_id])):
            b_spam = gaussian_pdf(emails[mail_id][feature_id], mu_log[0][feature_id], simg_log[0][feature_id])
            a_spam = a_spam * b_spam

            b_not_spam = gaussian_pdf(emails[mail_id][feature_id], mu_log[1][feature_id], simg_log[1][feature_id])
            a_not_spam = a_not_spam * b_not_spam

        p_pred_spam[mail_id] = pi_c[0] * a_spam
        p_pred_not_spam[mail_id] = pi_c[1] * a_not_spam

        if p_pred_spam[mail_id] > p_pred_not_spam[mail_id]:
            y_pred = 1
        else:
            y_pred = 0

        if y_pred != emails_label[mail_id]:
            error_counter += 1

        error_rate = error_counter / len(emails_label)

    return error_rate


error_x_train_znorm = classify_error_znorm(x_train_znorm, y_train)
error_x_test_znorm = classify_error_znorm(x_test_znorm, y_test)
print("error rate for z-normalization on x_train: ", error_x_train_znorm)
print("error rate for z-normalization on x_test: ", error_x_test_znorm)

print()

error_x_train_logtrans = classify_error_logtrans(x_train_logtrans, y_train)
error_x_test_logtrans = classify_error_logtrans(x_test_logtrans, y_test)
print("error rate for log-transform on x_train: ", error_x_train_logtrans)
print("error rate for log-transform on x_test: ", error_x_test_logtrans)
