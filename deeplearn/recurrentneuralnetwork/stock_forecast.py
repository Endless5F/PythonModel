# -*-coding:utf-8-*-
# 根据股价，预测股票趋势，仅使用股价一个维度预测，根据前n天的股价数据预测n+1天的
# RNN预测股价实战summary：
# 1、通过搭建RNN模型，实现了基于历史数据对次日股价的预测；
# 2、熟悉了RNN模型的数据格式结构；
# 3、掌握了数字序列的数据预处理方法；
# 4、实现了预测数据存储，通过可视化局部细节了解了RNN用于股价预测的局限性：信息延迟
# 5、RNN模型参考资料：https://keras.io/layers/recurrent/

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

data = pd.read_csv('./csv/zgpa_train.csv')
price = data.loc[:, 'close']

# 归一化处理
price_norm = price / max(price)

plt.figure()
plt.plot(price)
plt.title('close price')
plt.xlabel('time')
plt.ylabel('price')


# plt.show()

def extract_data(data, time_step):
    X = []  # List
    y = []  # List
    for i in range(len(data) - time_step):
        X.append([a for a in data[i:i + time_step]])
        y.append(data[i + time_step])
    X = np.array(X)  # List转二维数组，每5个在一个数组中
    X = X.reshape(X.shape[0], X.shape[1], 1)  # reshape成三维数组，一维所有数组，二维内部有5个三维数组，三维数组中只有一个值
    print(X)
    y = np.array(y)
    return X, y


time_step = 8
X, y = extract_data(price_norm, time_step)
print(X.shape, len(y))

model = Sequential()
model.add(SimpleRNN(units=5, input_shape=(time_step, 1), activation='relu'))
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
model.fit(X, y, batch_size=30, epochs=200)

# 训练集的y
y_train = [i * max(price) for i in y]
# 预测集的y
y_predict = model.predict(X) * max(price)

plt.figure()
plt.plot(y_train, label='real price')
plt.plot(y_predict, label='predict price')
plt.title('close price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()

# 预测测试数据
# 训练一次以后发现预测出来的结果不理想，很可能是模型进行初始化的时候选取的随机系数不合适，导致梯度下降搜索时遇到了局部极小值
# 解决办法：尝试再次建立模型并训练
# 多层感知机结构在进行模型求解时，会给定一组随机的初始化权重系数，这种情况是正常的。通常我们可以观察损失函数是否在变小来发现模型求解是否正常
data_test = pd.read_csv('./csv/zgpa_test.csv')
price_test = data_test.loc[:, 'close']
price_test.head()
price_test_norm = price_test / max(price)
# extract X_test and y_test
X_test_norm, y_test_norm = extract_data(price_test_norm, time_step)
print(X_test_norm.shape, len(y_test_norm))
y_test_predict = model.predict(X_test_norm) * max(price)
y_test = [i * max(price) for i in y_test_norm]

plt.figure()
plt.plot(y_test, label='real price_test')
plt.plot(y_test_predict, label='predict price_test')
plt.title('close price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()

# 保存测试结果
result_y_test = np.array(y_test).reshape(-1, 1)
result_y_test_predict = y_test_predict
print(result_y_test.shape, result_y_test_predict.shape)
# 合并两个数组，按照列合并
result = np.concatenate((result_y_test, result_y_test_predict), axis=1)
print(result.shape)
result = pd.DataFrame(result, columns=['real_price_test', 'predict_price_test'])
result.to_csv('./csv/zgpa_predict_test.csv')
