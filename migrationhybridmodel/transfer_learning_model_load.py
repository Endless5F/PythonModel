# -*-coding:utf-8-*-
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model

model = load_model('./transfer_learning_model.h5')

data = pd.read_csv('./csv/transfer_data.csv')
X = data.loc[:, 'x']
y = data.loc[:, 'y']


data2 = pd.read_csv('./csv/transfer_data2.csv')
X2 = data2.loc[:, 'x2']
y2 = data2.loc[:, 'y2']
X2 = np.array(X2).reshape(-1, 1)
y2_predict = model.predict(X2)
fig5 = plt.figure(figsize=(7, 5))
plt.scatter(X, y, label='data1')
plt.scatter(X2, y2, label='data2')
plt.plot(X, y2_predict, 'r', label='predict2')
plt.title('y vs x')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

model.fit(X2, y2, epochs=40)
y2_predict = model.predict(X2)
fig6 = plt.figure(figsize=(7, 5))
plt.scatter(X, y, label='data1')
plt.scatter(X2, y2, label='data2')
plt.plot(X, y2_predict, 'r', label='predict2')
plt.title('y vs x(epochs=10)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
fig8 = plt.figure(figsize=(7, 5))
plt.scatter(X, y, label='data1')
plt.scatter(X2, y2, label='data2')
plt.plot(X, y2_predict, 'r', label='predict2')
plt.title('y vs x(epochs=40)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
