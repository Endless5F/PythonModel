# -*-coding:utf-8-*-

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

data = pd.read_csv('./csv/data.csv')
X = data.drop(['y'], axis=1)
y = data.loc[:, 'y']

plt.figure()
plt.scatter(data.loc[:, 'x1'][y == 0], data.loc[:, 'x2'][y == 0])
plt.scatter(data.loc[:, 'x1'][y == 1], data.loc[:, 'x2'][y == 1])
plt.title('x1 - x2')
plt.xlabel('x1')
plt.ylabel('x2')
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)
mlp = Sequential()
mlp.add(Dense(units=20, input_dim=2, activation='sigmoid'))
mlp.add(Dense(units=1, activation='sigmoid'))
mlp.summary()

mlp.compile(optimizer='adam', loss='binary_crossentropy')
mlp.fit(X_train, y_train, epochs=3000)

y_train_predict = mlp.predict_classes(X_train)
print(accuracy_score(y_train, y_train_predict))

y_test_predict = mlp.predict_classes(X_test)
print(accuracy_score(y_test, y_test_predict))

# meshgrid:生成网格点坐标矩阵  array:生成数组
xx, yy = np.meshgrid(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))

# c_:是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等
# numpy中扁平化函数ravel()和flatten()的区别,在使用过程中flatten()分配了新的内存,但ravel()返回的是一个数组的视图
X_range = np.c_[xx.ravel(), yy.ravel()]
y_range_predict = mlp.predict_classes(X_range)

# Series 是带标签的一维数组
# 列表解析List Comprehensions: i[0] for i in y_range_predict
y_range_predict_form = pd.Series(i[0] for i in y_range_predict)

plt.figure()
plt.scatter(data.loc[:, 'x1'][y == 0], data.loc[:, 'x2'][y == 0])
plt.scatter(data.loc[:, 'x1'][y == 1], data.loc[:, 'x2'][y == 1])
plt.scatter(X_range[:, 0][y_range_predict_form == 0], X_range[:, 1][y_range_predict_form == 0])
plt.scatter(X_range[:, 0][y_range_predict_form == 1], X_range[:, 1][y_range_predict_form == 1])
plt.title('x1_range - x2_range')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()