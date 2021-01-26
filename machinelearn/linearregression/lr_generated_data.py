# 单因子线性回归实战

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 读取csv文件
data = pd.read_csv("./csv/generated_data.csv")
# print(type(data), data.shape)

# 读取csv文件 x，y列
# 　X[:,0] 二维数组取第１维所有数据
x = data.loc[:, 'x']
y = data.loc[:, 'y']
# print(x, y)
print(type(y), y.shape)

# 绘制源数据点图
# plt.figure(figsize=(10, 10))
# plt.scatter(x, y)
# plt.show()

# 一维转二维
x = np.array(x)
x = x.reshape(-1, 1)

lr_mode = LinearRegression()
# 参数需要二维数组
lr_mode.fit(x, y)

# 测试预测结果
y_predict = lr_mode.predict(x)
# print(y_predict, y)

# 预测新数据
y_3 = lr_mode.predict([[3.5]])
print(y_3)

a = lr_mode.coef_
b = lr_mode.intercept_
print(a, b)

MSE = mean_squared_error(y, y_predict)
R2 = r2_score(y, y_predict)
print(MSE, R2)

plt.figure()
plt.scatter(y, y_predict)
plt.show()
