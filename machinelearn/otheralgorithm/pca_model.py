# -*-coding:utf-8-*-
# PCA实战task：
# 1、基于iris_data.csv数据，建立KNN模型实现数据分类（n_neighbors=3）
# 2、对数据进行标准化处理，选取一个维度可视化处理后的效果
# 3、进行与原数据等维度PCA，查看各主成分的方差比例
# 4、保留合适的主成分，可视化降维后的数据
# 5、基于降维后数据建立KNN模型，与原数据表现进行对比

# 归一化：１）把数据变成(０，１)或者（1,1）之间的小数。主要是为了数据处理方便提出来的，把数据映射到0～1范围之内处理，更加便捷快速。２）把有量纲表达式变成无量纲表达式，便于不同单位或量级的指标能够进行比较和加权。归一化是一种简化计算的方式，即将有量纲的表达式，经过变换，化为无量纲的表达式，成为纯量。
# 标准化：在机器学习中，我们可能要处理不同种类的资料，例如，音讯和图片上的像素值，这些资料可能是高维度的，资料标准化后会使每个特征中的数值平均变为0(将每个特征的值都减掉原始资料中该特征的平均)、标准差变为1，这个方法被广泛的使用在许多机器学习算法中(例如：支持向量机、逻辑回归和类神经网络)。
# 中心化：平均值为0，对标准差无要求
# 归一化和标准化的区别：归一化是将样本的特征值转换到同一量纲下把数据映射到[0,1]或者[-1, 1]区间内，仅由变量的极值决定，因区间放缩法是归一化的一种。标准化是依照特征矩阵的列处理数据，其通过求z-score的方法，转换为标准正态分布，和整体样本分布相关，每个样本点都能对标准化产生影响。它们的相同点在于都能取消由于量纲不同引起的误差；都是一种线性变换，都是对向量X按照比例压缩再进行平移。
# 标准化和中心化的区别：标准化是原始分数减去平均数然后除以标准差，中心化是原始分数减去平均数。 所以一般流程为先中心化再标准化。
# 无量纲：我的理解就是通过某种方法能去掉实际过程中的单位，从而简化计算。


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

data = pd.read_csv('./csv/iris_data.csv')
X = data.drop(['target', 'label'], axis=1)
y = data.loc[:, 'label']

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
y_predict = knn.predict(X)
print(accuracy_score(y, y_predict))

# 拟合变换, 去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本。
X_normal = StandardScaler().fit_transform(X)

x1_mean = X.loc[:, 'sepal length'].mean()
x1_norm_mean = X_normal[:, 0].mean()
x1_sigma = X.loc[:, 'sepal length'].std()
x1_norm_sigma = X_normal[:, 0].std()
print(x1_mean, x1_sigma, x1_norm_mean, x1_norm_sigma)

plt.figure()
plt.subplot(121)
plt.hist(data.loc[:, 'sepal width'], bins=100)
plt.subplot(122)
plt.hist(X_normal[:, 0], bins=100)
plt.show()

# 使用原维度维数，查看各维度方差比，可决定降低的维度
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_normal)
# 解释方差比
var_ratio = pca.explained_variance_ratio_
print(var_ratio)

# 输出结果: [0.72770452 0.23030523 0.03683832 0.00515193]
# 可以看出前两个方差比较大，后两个方差很小，即影响结果小。因此前两个属性应该作为主成分分析的维度

plt.figure()
# 竖值条形图
plt.bar([1, 2, 3, 4], var_ratio)
# 条形图 每个item的标注显示名称
plt.xticks([1, 2, 3, 4], ['PC1', 'PC2', 'PC3', 'PC4'])
plt.show()

# 降低两个维度
pca2 = PCA(n_components=2)
X_pca_2 = pca.fit_transform(X_normal)
plt.figure()
setosa = plt.scatter(X_pca_2[:, 0][y == 0], X_pca_2[:, 1][y == 0])
versicolor = plt.scatter(X_pca_2[:, 0][y == 1], X_pca_2[:, 1][y == 1])
virginica = plt.scatter(X_pca_2[:, 0][y == 2], X_pca_2[:, 1][y == 2])
plt.legend((setosa, versicolor, virginica), ('setosa', 'versicolor', 'virginica'))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# 通过降低后的维度，再次使用knn预测准确率
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_pca_2, y)
y_predict_pca = knn.predict(X_pca_2)
print(accuracy_score(y, y_predict_pca))