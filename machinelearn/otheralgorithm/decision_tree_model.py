# -*-coding:utf-8-*-

import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt

data = pd.read_csv('./csv/iris_data.csv')

X = data.drop(['target', 'label'], axis=1)
y = data.loc[:, 'label']
print(X.shape, y.shape)

# criterion标准: entropy熵 ID3
dc_tree = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=5)
dc_tree.fit(X, y)

y_predict = dc_tree.predict(X)
accuracy = accuracy_score(y, y_predict)
print(accuracy)

tree.plot_tree(dc_tree, filled='True', feature_names=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'],
               class_names=['setosa', 'versicolor', 'virginica'])
plt.show()
