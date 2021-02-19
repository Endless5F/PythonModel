# -*-coding:utf-8-*-
# 手写数字图片识别，多分类

# mnist.load_data()报错，出现错误的原因是因为无法连接国外的那个下载链接
# from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

# 由于mnist.load_data()报错。因此需要把mnist.npz下载到本地，并通过numpy加载
data = np.load("../../data/mnist.npz")
X_train, y_train, X_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
print(type(X_train), X_train.shape)

img1 = X_train[0]
plt.figure()
plt.imshow(img1)
plt.title(y_train[0])
# plt.show()

# shape 读取矩阵长度
feature_size = img1.shape[0] * img1.shape[1]
# X_train 是三维数组，一维大小为60000，二维和三维大小都是28，reshape(60000,784) 即将二三维打平
X_train_format = X_train.reshape(X_train.shape[0], feature_size)
print(type(X_train_format), X_train_format.shape)
X_test_format = X_test.reshape(X_test.shape[0], feature_size)

# 归一化数据，由于是颜色值 0-255，因此统一除去255
X_train_normal = X_train_format / 255
X_test_normal = X_test_format / 255

# 将类别标签向量转换为二进制（只有0和1）的矩阵类型表示。每一个标签用矩阵的对应的行向量来表示。
y_train_format = to_categorical(y_train)
y_test_format = to_categorical(y_test)

# Sigmoid函数实际上就是把数据映射到一个(0,1)的空间上，也就是说，Sigmoid函数如果用来分类的话，只能进行二分类，
# 而这里的softmax函数可以看做是Sigmoid函数的一般化，可以进行多分类。softmax函数的函数表达式为：σ(z)[j]=e^Z[j] / (∑k=1-K e^Z[k]) []代表角标
mlp = Sequential()
mlp.add(Dense(units=392, activation='sigmoid', input_dim=feature_size))
mlp.add(Dense(units=392, activation='sigmoid'))
mlp.add(Dense(units=10, activation='softmax'))
mlp.summary()
# loss 目标函数，或称损失函数
mlp.compile(optimizer='adam', loss='binary_crossentropy')
mlp.fit(X_train_normal, y_train_format, epochs=10)
y_normal_predict = mlp.predict_classes(X_train_normal)
print(accuracy_score(y_train, y_normal_predict))

y_test_predict = mlp.predict_classes(X_test_normal)
print(accuracy_score(y_test, y_test_predict))

img2 = X_test[100]
fig2 = plt.figure(figsize=(3, 3))
plt.imshow(img2)
plt.title(y_test_predict[100])
plt.show()
