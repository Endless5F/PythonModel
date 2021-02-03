# -*-coding:utf-8-*-
# 考试通过逻辑回归

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("./csv/examdata.csv")

plt.figure()
mask = data.loc[:, 'Pass'] == 1
passed = plt.scatter(data.loc[:, 'Exam1'][mask], data.loc[:, 'Exam2'][mask])
failed = plt.scatter(data.loc[:, 'Exam1'][~mask], data.loc[:, 'Exam2'][~mask])
plt.title('Exam1 - Exam2')
plt.xlabel('Exam1')
plt.ylabel('Exam2')
plt.legend((passed, failed), ('passed', 'failed'))
# plt.show()

X = data.drop(['Pass'], axis=1)
x1 = data.loc[:, 'Exam1']
x2 = data.loc[:, 'Exam2']
y = data.loc[:, 'Pass']

LR = LogisticRegression()
LR.fit(X, y)
y_predict = LR.predict(X)
print(y_predict)

theta0 = LR.intercept_
theta1, theta2 = LR.coef_[0][0], LR.coef_[0][1]

# 边界函数：  𝜃0+𝜃1𝑋1+𝜃2𝑋2=0
x2_new = -(theta0 + theta1 * x1) / theta2
plt.plot(x1, x2_new)
# plt.show()

accuracy = accuracy_score(y, y_predict)
print(accuracy)

y_test = LR.predict([[70, 65]])
print(y_test)

# 二阶边界函数： 𝜃0+𝜃1𝑋1+𝜃2𝑋2+𝜃3𝑋21+𝜃4𝑋22+𝜃5𝑋1𝑋2=0
x1_2 = x1 * x1
x2_2 = x2 * x2
x1_x2 = x1 * x2
X_new = {'X1': x1, 'X2': x2, 'X1_2': x1_2, 'X2_2': x2_2, 'X1_X2': x1_x2}
X_new = pd.DataFrame(X_new)
print(X_new)

LR2 = LogisticRegression()
LR2.fit(X_new, y)
y2_predict = LR2.predict(X_new)
accuracy2 = accuracy_score(y, y2_predict)
print(accuracy2)

# 从小到大排序，防止画曲线图交叉效果
x1_new = x1.sort_values()
print(x1, x1_new)

theta0 = LR2.intercept_
theta1, theta2, theta3, theta4, theta5 = LR2.coef_[0][0], LR2.coef_[0][1], LR2.coef_[0][2], LR2.coef_[0][3], \
                                         LR2.coef_[0][4]

# 𝑎𝑥2+𝑏𝑥+𝑐=0:𝑥1=(−𝑏+(𝑏2−4𝑎𝑐).5)/2𝑎,𝑥1=(−𝑏−(𝑏2−4𝑎𝑐).5)/2𝑎  𝜃4𝑋22+(𝜃5𝑋1++𝜃2)𝑋2+(𝜃0+𝜃1𝑋1+𝜃3𝑋21)=0
a = theta4
b = theta5 * x1_new + theta2
c = theta0 + theta1 * x1_new + theta3 * x1_new * x1_new
X2_new_boundary = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
plt.plot(x1_new, X2_new_boundary)
plt.show()
