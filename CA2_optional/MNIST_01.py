#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/12 17:37
# @File    : MNIST_01.py
# @Author  : NusLuoKe

import os
import time

from CA2_optional import util, my_models

# load MNIST data and do pre-works
(x_train, y_train), (x_test, y_test) = util.load_data()
print(x_train.shape)

batch_size = 32
nb_epoch = 1

model = my_models.cnn01()
# print the model structure
print(model.summary())

# train the model
start = time.time()
h = model.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size, validation_data=(x_test, y_test),
              shuffle=True, verbose=2, )

# save the model to the following directory
model_dir = 'S:/MNIST_MODELS/model_01'
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
model.save(model_dir + '/MNIST_MODEL_01.h5')

end = time.time()
# print the over all time from training start to the model saved
print('@ Total Time Spent: %.2f seconds' % (end - start))

# plot figures of accuracy and loss of every epoch and a visible test result
util.plot_acc_loss(h, nb_epoch)

# print loss and accuracy on the whole training set and test set
loss, accuracy = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy = %.2f %%     loss = %f" % (accuracy * 100, loss))
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy = %.2f %%    loss = %f" % (accuracy * 100, loss))
