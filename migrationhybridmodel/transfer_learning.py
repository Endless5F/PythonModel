# -*-coding:utf-8-*-
# 任务：基于transfer_data.csv数据，建立mlp模型，再实现模型迁移学习：
# 1.实现x对y的预测，可视化结果
# 2.基于新数据transfer_data2.csv，对前模型进行二次训练，对比模型训练次数少的情况下的表现
# 备注：模型结构：mlp，两个隐藏层，每层50个神经元，激活函数relu，输出层激活函数linear，迭代次数：100次

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('./csv/transfer_data.csv')
X = data.loc[:, 'x']
y = data.loc[:, 'y']

plt.figure()
plt.scatter(X, y)
plt.title('y vs x')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 数据集变成一列
X = np.array(X).reshape(-1, 1)

model = Sequential()
model.add(Dense(units=50, input_dim=1, activation='relu'))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
model.fit(X, y, epochs=200)

y_predict = model.predict(X)
plt.figure(figsize=(7, 5))
plt.scatter(X, y)
plt.plot(X, y_predict, 'r')
plt.title('y vs x(epochs=100)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

model.save("transfer_learning_model.h5")
