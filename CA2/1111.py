#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/11 19:20
# @File    : 1111.py
# @Author  : NusLuoKe

from sklearn.neighbors import KNeighborsClassifier

X = [[1, 2], [2, 3], [5, 6]]
y = [[2, 3], [6, 7], [2, 4]]

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, y)

print(neigh.predict([[1.1, 2,1]]))
