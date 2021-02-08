# -*-coding:utf-8-*-
# 好坏质检分类实战task：
# 1、基于data_class_raw.csv数据，根据高斯分布概率密度函数，寻找异常点并剔除
# 2、基于data_class_processed.csv数据，进行PCA处理，确定重要数据维度及成分
# 3、完成数据分离，数据分离参数：random_state=4,test_size=0.4
# 4、建立KNN模型完成分类，n_neighbors取10，计算分类准确率，可视化分类边界
# 5、计算测试数据集对应的混淆矩阵，计算准确率、召回率、特异度、精确率、F1分数
# 6、尝试不同的n_neighbors（1-20）,计算其在训练数据集、测试数据集上的准确率并作图

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv('./csv/data_class_raw.csv')
X = data.drop(['y'], axis=1)
y = data.loc[:, 'y']

fig1 = plt.figure(figsize=(5, 5))
bad = plt.scatter(X.loc[:, 'x1'][y == 0], X.loc[:, 'x2'][y == 0])
good = plt.scatter(X.loc[:, 'x1'][y == 1], X.loc[:, 'x2'][y == 1])
plt.legend((good, bad), ('good', 'bad'))
plt.title('raw data')
plt.xlabel('x1')
plt.ylabel('x2')
# plt.show()

# EllipticEnvelope 异常检测
ad_model = EllipticEnvelope(contamination=0.02)
ad_model.fit(X[y == 0])
y_predict_bad = ad_model.predict(X[y == 0])

plt.figure()
plt.scatter(X.loc[:, 'x1'][y == 0], X.loc[:, 'x2'][y == 0])
plt.scatter(X.loc[:, 'x1'][y == 1], X.loc[:, 'x2'][y == 1])
plt.scatter(X.loc[:, 'x1'][y == 0][y_predict_bad == -1], X.loc[:, 'x2'][y == 0][y_predict_bad == -1], marker='x', s=150)
plt.title('EllipticEnvelope data')
plt.xlabel('x1')
plt.ylabel('x2')
# plt.show()

data = pd.read_csv('./csv/data_class_processed.csv')
data.head()
X = data.drop(['y'], axis=1)
y = data.loc[:, 'y']

# 标准化处理 PCA尝试降维处理
X_norm = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_norm)
ratio = pca.explained_variance_ratio_
print(ratio)
# 输出方差比：[0.5369408 0.4630592] 结果两个维度占比都比较高无法降维处理

# 数据分离
# test_size：测试数据占样本数据的比例，若整数则样本数量
# random_state：设置随机数种子，保证每次都是同一个随机数。若为0或不填，则每次得到数据都不一样
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4, test_size=0.4)

knn_10 = KNeighborsClassifier(n_neighbors=10)
knn_10.fit(X_train, y_train)
y_train_predict = knn_10.predict(X_train)
y_test_predict = knn_10.predict(X_test)

accuracy_train = accuracy_score(y_train, y_train_predict)
accuracy_test = accuracy_score(y_test, y_test_predict)
print(accuracy_train, accuracy_test)

# 构建虚拟数据 并预测，可视化分类边界
xx, yy = np.meshgrid(np.arange(0, 10, 0.05), np.arange(0, 10, 0.05))
x_range = np.c_[xx.ravel(), yy.ravel()]
print(x_range.shape)
y_range_predict = knn_10.predict(x_range)
fig4 = plt.figure(figsize=(10, 10))
knn_bad = plt.scatter(x_range[:, 0][y_range_predict == 0], x_range[:, 1][y_range_predict == 0])
knn_good = plt.scatter(x_range[:, 0][y_range_predict == 1], x_range[:, 1][y_range_predict == 1])
bad = plt.scatter(X.loc[:, 'x1'][y == 0], X.loc[:, 'x2'][y == 0])
good = plt.scatter(X.loc[:, 'x1'][y == 1], X.loc[:, 'x2'][y == 1])
plt.legend((good, bad, knn_good, knn_bad), ('good', 'bad', 'knn_good', 'knn_bad'))
plt.title('prediction result')
plt.xlabel('x1')
plt.ylabel('x2')
# plt.show()

cm = confusion_matrix(y_test, y_test_predict)

TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
print(TP, TN, FP, FN)

# 准确率: 整体样本中，预测正确样本数的比例  Accuracy = (TP + TN)/(TP + TN + FP + FN)
accuracy = (TP + TN) / (TP + TN + FP + FN)
print(accuracy)

# 灵敏度（召回率）: 正样本中，预测正确的比例  Sensitivity = Recall = TP/(TP + FN)
recall = TP / (TP + FN)
print(recall)

# 特异度: 负样本中，预测正确的比例  Specificity = TN/(TN + FP)
specificity = TN / (TN + FP)
print(specificity)

# 精确率: 预测结果为正的样本中，预测正确的比例  Precision = TP/(TP + FP)
precision = TP / (TP + FP)
print(precision)

# F1分数: 综合Precision和Recall的一个判断指标 F1 Score = 2*Precision X Recall/(Precision + Recall)
f1 = 2 * precision * recall / (precision + recall)
print(f1)

# 尝试不同的n_neighbors（1-20）,计算其在训练数据集、测试数据集上的准确率并作图
n = [i for i in range(1, 21)]
accuracy_trains = []
accuracy_tests = []
for i in n:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_train_predict_i = knn.predict(X_train)
    y_test_predict_i = knn.predict(X_test)
    accuracy_train_i = accuracy_score(y_train, y_train_predict_i)
    accuracy_test_i = accuracy_score(y_test, y_test_predict_i)
    accuracy_trains.append(accuracy_train_i)
    accuracy_tests.append(accuracy_test_i)

print(accuracy_trains, accuracy_tests)

plt.figure()
plt.subplot(121)
plt.plot(n, accuracy_trains, marker='o')
plt.title('training accuracy vs n_neighbors')
plt.xlabel('n_neighbors')
plt.ylabel('accuracy')
plt.subplot(122)
plt.plot(n, accuracy_tests, marker='o')
plt.title('testing accuracy vs n_neighbors')
plt.xlabel('n_neighbors')
plt.ylabel('accuracy')
plt.show()