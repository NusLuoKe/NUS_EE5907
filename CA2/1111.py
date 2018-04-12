#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/11 19:20
# @File    : 1111.py
# @Author  : NusLuoKe


##############################visulize the projection#########################################
img_0_train_2d, img_1_train_2d, img_2_train_2d, img_3_train_2d = [], [], [], []
img_4_train_2d, img_5_train_2d, img_6_train_2d = [], [], []
img_7_train_2d, img_8_train_2d, img_9_train_2d = [], [], []

img_0_train_3d, img_1_train_3d, img_2_train_3d, img_3_train_3d = [], [], [], []
img_4_train_3d, img_5_train_3d, img_6_train_3d = [], [], []
img_7_train_3d, img_8_train_3d, img_9_train_3d = [], [], []
for i in range(60000):
    if y_train[i] == 0:
        img_0_train_2d.append(n2_low_dim_data[i])
        img_0_train_3d.append(n3_low_dim_data[i])
    elif y_train[i] == 1:
        img_1_train_2d.append(n2_low_dim_data[i])
        img_1_train_3d.append(n3_low_dim_data[i])
    elif y_train[i] == 2:
        img_2_train_2d.append(n2_low_dim_data[i])
        img_2_train_3d.append(n3_low_dim_data[i])
    elif y_train[i] == 3:
        img_3_train_2d.append(n2_low_dim_data[i])
        img_3_train_3d.append(n3_low_dim_data[i])
    elif y_train[i] == 4:
        img_4_train_2d.append(n2_low_dim_data[i])
        img_4_train_3d.append(n3_low_dim_data[i])
    elif y_train[i] == 5:
        img_5_train_2d.append(n2_low_dim_data[i])
        img_5_train_3d.append(n3_low_dim_data[i])
    elif y_train[i] == 6:
        img_6_train_2d.append(n2_low_dim_data[i])
        img_6_train_3d.append(n3_low_dim_data[i])
    elif y_train[i] == 7:
        img_7_train_2d.append(n2_low_dim_data[i])
        img_7_train_3d.append(n3_low_dim_data[i])
    elif y_train[i] == 8:
        img_8_train_2d.append(n2_low_dim_data[i])
        img_8_train_3d.append(n3_low_dim_data[i])
    else:
        img_9_train_2d.append(n2_low_dim_data[i])
        img_9_train_3d.append(n3_low_dim_data[i])

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