# -*-coding:utf-8-*-
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

data = pd.read_csv("./csv/data.csv")

X = data.drop(['labels'], axis=1)
y = data.loc[:, 'labels']

# 分类前
plt.figure()
plt.scatter(data.loc[:, 'V1'], data.loc[:, 'V2'])
plt.title('V1 - V2')
plt.xlabel('V1')
plt.ylabel('V2')

# 分类后
plt.figure()
label0 = plt.scatter(data.loc[:, 'V1'][y == 0], data.loc[:, 'V2'][y == 0])
label1 = plt.scatter(data.loc[:, 'V1'][y == 1], data.loc[:, 'V2'][y == 1])
label2 = plt.scatter(data.loc[:, 'V1'][y == 2], data.loc[:, 'V2'][y == 2])
plt.title('V1 - V2')
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend((label0, label1, label2), ("label0", "label1", "label2"))
plt.show()

# n_clusters == k
km = KMeans(n_clusters=3, random_state=0)
km.fit(X)

y_predict = km.predict(X)
accuracy = accuracy_score(y, y_predict)
print(accuracy)

# 中心点
centers = km.cluster_centers_
plt.subplot(121)
label0 = plt.scatter(data.loc[:, 'V1'][y == 0], data.loc[:, 'V2'][y == 0])
label1 = plt.scatter(data.loc[:, 'V1'][y == 1], data.loc[:, 'V2'][y == 1])
label2 = plt.scatter(data.loc[:, 'V1'][y == 2], data.loc[:, 'V2'][y == 2])
plt.title('KMeans: V1 - V2')
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend((label0, label1, label2), ("label0", "label1", "label2"))
plt.scatter(centers[:, 0], centers[:, 1])

# 预测后分类：如果预测后正确率较低，可以重新画出来看下原因
plt.subplot(122)
label0 = plt.scatter(data.loc[:, 'V1'][y_predict == 0], data.loc[:, 'V2'][y_predict == 0])
label1 = plt.scatter(data.loc[:, 'V1'][y_predict == 1], data.loc[:, 'V2'][y_predict == 1])
label2 = plt.scatter(data.loc[:, 'V1'][y_predict == 2], data.loc[:, 'V2'][y_predict == 2])
plt.title('y_predict: V1 - V2')
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend((label0, label1, label2), ("label0", "label1", "label2"))
plt.scatter(centers[:, 0], centers[:, 1])
plt.show()

# 训练后 需要校正，原因：原分类1 可能对应预测分类2
y_corrected = []
for i in y_predict:
    if i == 0:
        y_corrected.append(1)
    elif i == 1:
        y_corrected.append(2)
    else:
        y_corrected.append(0)
print(pd.value_counts(y_corrected), pd.value_counts(y))

print(accuracy_score(y, y_corrected))

y_corrected = np.array(y_corrected)
print(type(y_corrected))
plt.subplot(121)
label0 = plt.scatter(X.loc[:, 'V1'][y_corrected == 0], X.loc[:, 'V2'][y_corrected == 0])
label1 = plt.scatter(X.loc[:, 'V1'][y_corrected == 1], X.loc[:, 'V2'][y_corrected == 1])
label2 = plt.scatter(X.loc[:, 'V1'][y_corrected == 2], X.loc[:, 'V2'][y_corrected == 2])
plt.title("corrected data")
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend((label0, label1, label2), ('label0', 'label1', 'label2'))
plt.scatter(centers[:, 0], centers[:, 1])

plt.subplot(122)
label0 = plt.scatter(X.loc[:, 'V1'][y == 0], X.loc[:, 'V2'][y == 0])
label1 = plt.scatter(X.loc[:, 'V1'][y == 1], X.loc[:, 'V2'][y == 1])
label2 = plt.scatter(X.loc[:, 'V1'][y == 2], X.loc[:, 'V2'][y == 2])
plt.title("labled data")
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend((label0, label1, label2), ('label0', 'label1', 'label2'))
plt.scatter(centers[:, 0], centers[:, 1])
plt.show()
