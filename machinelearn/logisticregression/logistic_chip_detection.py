# -*-coding:utf-8-*-
# 芯片检测逻辑回归

# load the data
import pandas as pd
import numpy as np

data = pd.read_csv('./csv/chip_test.csv')

# add label mask
mask = data.loc[:, 'pass'] == 1

# visualize the data
from matplotlib import pyplot as plt

fig1 = plt.figure()
passed = plt.scatter(data.loc[:, 'test1'][mask], data.loc[:, 'test2'][mask])
failed = plt.scatter(data.loc[:, 'test1'][~mask], data.loc[:, 'test2'][~mask])
plt.title('test1-test2')
plt.xlabel('test1')
plt.ylabel('test2')
plt.legend((passed, failed), ('passed', 'failed'))

X = data.drop(['pass'], axis=1)
y = data.loc[:, 'pass']
X1 = data.loc[:, 'test1']
X2 = data.loc[:, 'test2']
X1.head()
# create new data
X1_2 = X1 * X1
X2_2 = X2 * X2
X1_X2 = X1 * X2
# 利用字典创建DataFrame
X_new = {'X1': X1, 'X2': X2, 'X1_2': X1_2, 'X2_2': X2_2, 'X1_X2': X1_X2}
X_new = pd.DataFrame(X_new)
print(X_new)

# establish the model and train it
from sklearn.linear_model import LogisticRegression

LR2 = LogisticRegression()
LR2.fit(X_new, y)

from sklearn.metrics import accuracy_score

y2_predict = LR2.predict(X_new)
accuracy2 = accuracy_score(y, y2_predict)
print(accuracy2)

X1_new = X1.sort_values()
theta0 = LR2.intercept_
theta1, theta2, theta3, theta4, theta5 = LR2.coef_[0][0], LR2.coef_[0][1], LR2.coef_[0][2], LR2.coef_[0][3], \
                                         LR2.coef_[0][4]

a = theta4
b = theta5 * X1_new + theta2
c = theta0 + theta1 * X1_new + theta3 * X1_new * X1_new
X2_new_boundary = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
# 只画了一条曲线, 少了 (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a) 的情况
fig2 = plt.figure()
passed = plt.scatter(data.loc[:, 'test1'][mask], data.loc[:, 'test2'][mask])
failed = plt.scatter(data.loc[:, 'test1'][~mask], data.loc[:, 'test2'][~mask])
plt.plot(X1_new, X2_new_boundary)
plt.title('test1-test2')
plt.xlabel('test1')
plt.ylabel('test2')
plt.legend((passed, failed), ('passed', 'failed'))


# define f(x) 看似 x1、x2 是二阶函数，不过由于pass已知，因此就可以假设x1为已知值，x2为未知值，即可转化为一阶函数
def f(x):
    a = theta4
    b = theta5 * x + theta2
    c = theta0 + theta1 * x + theta3 * x * x
    # 一元二次方程求根公式：x1=（-b+（b^2-4ac)^1/2）/2a ,x2=（-b-（b^2-4ac)^1/2）/2a
    boundary1 = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    boundary2 = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
    return boundary1, boundary2


X2_new_boundary1 = []
X2_new_boundary2 = []
for x in X1_new:
    X2_new_boundary1.append(f(x)[0])
    X2_new_boundary2.append(f(x)[1])
print(X2_new_boundary1, X2_new_boundary2)
# 点不全所以无法封闭
fig3 = plt.figure()
passed = plt.scatter(data.loc[:, 'test1'][mask], data.loc[:, 'test2'][mask])
failed = plt.scatter(data.loc[:, 'test1'][~mask], data.loc[:, 'test2'][~mask])
plt.plot(X1_new, X2_new_boundary1)
plt.plot(X1_new, X2_new_boundary2)
plt.title('test1-test2')
plt.xlabel('test1')
plt.ylabel('test2')
plt.legend((passed, failed), ('passed', 'failed'))

# 自己根据方程模拟点位
X1_range = [-0.9 + x / 10000 for x in range(0, 19000)]
X1_range = np.array(X1_range)
X2_new_boundary1 = []
X2_new_boundary2 = []
for x in X1_range:
    X2_new_boundary1.append(f(x)[0])
    X2_new_boundary2.append(f(x)[1])

import matplotlib as mlp

mlp.rcParams['font.family'] = 'SimHei'
mlp.rcParams['axes.unicode_minus'] = False
fig4 = plt.figure()
passed = plt.scatter(data.loc[:, 'test1'][mask], data.loc[:, 'test2'][mask])
failed = plt.scatter(data.loc[:, 'test1'][~mask], data.loc[:, 'test2'][~mask])
plt.plot(X1_range, X2_new_boundary1, 'r')
plt.plot(X1_range, X2_new_boundary2, 'r')
plt.title('test1-test2')
plt.xlabel('测试1')
plt.ylabel('测试2')
plt.title('芯片质量预测')
plt.legend((passed, failed), ('passed', 'failed'))
plt.show()
