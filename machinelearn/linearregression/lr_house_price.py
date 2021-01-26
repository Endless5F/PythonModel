# 多因子线性回归实战

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("./csv/usa_housing_price.csv")
# print(data.head())

# 绘制源数据点图
plt.figure(figsize=(8, 8))
# 绘制源数据第一幅点图
plt.subplot(231)
plt.scatter(data.loc[:, "Avg. Area Income"], data.loc[:, "Price"])
plt.title("Price VS Income")
# 绘制源数据第二幅点图
plt.subplot(232)
plt.scatter(data.loc[:, "Avg. Area House Age"], data.loc[:, "Price"])
plt.title("Price VS House Age")
# 绘制源数据第三幅点图
plt.subplot(233)
plt.scatter(data.loc[:, "Avg. Area Number of Rooms"], data.loc[:, "Price"])
plt.title("Price VS Number of Rooms")
# 绘制源数据第四幅点图
plt.subplot(234)
plt.scatter(data.loc[:, "Area Population"], data.loc[:, "Price"])
plt.title("Price VS Population")
# 绘制源数据第五幅点图
plt.subplot(235)
plt.scatter(data.loc[:, "size"], data.loc[:, "Price"])
plt.title("Price VS size")
# plt.show()

y = data.loc[:, 'Price']

# drop: 删除行和列, 如果要删除某列，需要axis=1
x_multi = data.drop(["Price"], axis=1)
print(x_multi)

lr_multi = LinearRegression()
lr_multi.fit(x_multi, y)
y_predict_multi = lr_multi.predict(x_multi)
# print(y_predict_multi)

MSE = mean_squared_error(y, y_predict_multi)
R2 = r2_score(y, y_predict_multi)
print(MSE, R2)

plt.figure(figsize=(8, 8))
plt.scatter(y, y_predict_multi)
# plt.show()

# 预测
X_test = [65000, 5, 5, 30000, 200]
X_test = np.array(X_test).reshape(1, -1)
y_test_predict = lr_multi.predict(X_test)
print(y_test_predict)
