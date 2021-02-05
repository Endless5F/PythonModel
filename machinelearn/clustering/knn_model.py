# -*-coding:utf-8-*-

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("./csv/data.csv")

X = data.drop(['labels'], axis=1)
y = data.loc[:, 'labels']

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
y_predict = knn.predict(X)
print(accuracy_score(y, y_predict))

y_predict_knn_test = knn.predict([[80, 60]])
print(y_predict_knn_test)

print(pd.value_counts(y_predict), pd.value_counts(y))

plt.subplot(121)
label0 = plt.scatter(data.loc[:, 'V1'][y == 0], data.loc[:, 'V2'][y == 0])
label1 = plt.scatter(data.loc[:, 'V1'][y == 1], data.loc[:, 'V2'][y == 1])
label2 = plt.scatter(data.loc[:, 'V1'][y == 2], data.loc[:, 'V2'][y == 2])
plt.title('knn: V1 - V2')
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend((label0, label1, label2), ("label0", "label1", "label2"))

plt.subplot(122)
label0 = plt.scatter(data.loc[:, 'V1'][y_predict == 0], data.loc[:, 'V2'][y_predict == 0])
label1 = plt.scatter(data.loc[:, 'V1'][y_predict == 1], data.loc[:, 'V2'][y_predict == 1])
label2 = plt.scatter(data.loc[:, 'V1'][y_predict == 2], data.loc[:, 'V2'][y_predict == 2])
plt.title('y_predict: V1 - V2')
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend((label0, label1, label2), ("label0", "label1", "label2"))
plt.show()
