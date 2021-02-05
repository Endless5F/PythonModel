# -*-coding:utf-8-*-

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

data = pd.read_csv("./csv/data.csv")

X = data.drop(['labels'], axis=1)
y = data.loc[:, 'labels']
bw = estimate_bandwidth(X, n_samples=500)
ms = MeanShift(bandwidth=bw)
ms.fit(X)

y_predict = ms.predict(X)

plt.subplot(121)
label0 = plt.scatter(data.loc[:, 'V1'][y == 0], data.loc[:, 'V2'][y == 0])
label1 = plt.scatter(data.loc[:, 'V1'][y == 1], data.loc[:, 'V2'][y == 1])
label2 = plt.scatter(data.loc[:, 'V1'][y == 2], data.loc[:, 'V2'][y == 2])
plt.title('ms: V1 - V2')
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
