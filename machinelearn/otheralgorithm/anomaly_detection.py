# -*-coding:utf-8-*-
# 异常检测实战task

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.covariance import EllipticEnvelope

data = pd.read_csv('./csv/anomaly_data.csv')

fig1 = plt.figure(figsize=(10, 5))
plt.scatter(data.loc[:, 'x1'], data.loc[:, 'x2'])
plt.title('data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

x1 = data.loc[:, 'x1']
x2 = data.loc[:, 'x2']

fig2 = plt.figure(figsize=(10, 5))
plt.subplot(121)
# 绘制直方图
plt.hist(x1, bins=100)
plt.title('x1 Data distribution statistics')
plt.xlabel('x1')
plt.ylabel('The number of occurrences')
plt.subplot(122)
plt.hist(x2, bins=100)
plt.title('x2 distribution')
plt.xlabel('x2')
plt.ylabel('counts')
plt.show()

# 均值
x1_mean = x1.mean()
# 标准差
x1_sigma = x1.std()
x2_mean = x2.mean()
x2_sigma = x2.std()

# 生成0--20之间含有300个数的等间隔数列
x1_range = np.linspace(0, 20, 300)
# 正态概率密度函数
x1_normal = norm.pdf(x1_range, x1_mean, x1_sigma)
x2_range = np.linspace(0, 20, 300)
x2_normal = norm.pdf(x2_range, x2_mean, x2_sigma)

plt.figure()
plt.subplot(121)
# 绘制线条
plt.plot(x1_range, x1_normal)
plt.title('normal p(x1)')
plt.subplot(122)
plt.plot(x2_range, x2_normal)
plt.title('normal p(x2)')
plt.show()

ad_model = EllipticEnvelope()
ad_model.fit(data)

y_predict = ad_model.predict(data)
print(pd.value_counts(y_predict))

fig4 = plt.figure(figsize=(10, 6))
orginal_data = plt.scatter(data.loc[:, 'x1'], data.loc[:, 'x2'], marker='x')
anomaly_data = plt.scatter(data.loc[:, 'x1'][y_predict == -1], data.loc[:, 'x2'][y_predict == -1], marker='o',
                           facecolor='none', edgecolor='red', s=150)

plt.title('anomaly detection result')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend((orginal_data, anomaly_data), ('orginal_data', 'anomaly_data'))
# 为图形设置一些轴属性
# plt.axis([4.5, 15, 2.5, 15])
plt.show()
