# -*-coding:utf-8-*-
# 猫狗图片检测 VGG16

from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import ssl

# 防止报URL fetch failure错误，是由于vgg16模型无法下载导致,因此可直接下载后加载.h5模型文件
# Exception: URL fetch failure on https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5: None  -- [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1125)
ssl._create_default_https_context = ssl._create_unverified_context

# imagenet 表示加载imagenet与训练的神经网络权重
# VGG16 的框架是确定的, 而其权重参数的个数和结构完全由输入决定.
# 如果weight = None, 则输入尺寸可以任意指定,(范围不得小于48, 否则最后一个卷积层没有输出).
# 如果 weight = ‘imagenet’, 则输入尺寸必须严格等于(224,224), 权重的规模和结构有出入唯一决定, 使用了imagenet的权重,就必须使用训练时所对应的输入, 否则第一个全连接层的输入对接不上. (例如, 原来网络最后一个卷基层的输出为 300, 全连接层的神经元有1000个,则这里权重的结构为300X1000), 而其他的出入不能保证卷基层输出为300, 则对接不上会报错).
# 如果 weight = ‘path’, 则输入必须和path对应权值文件训练时的输入保持一致.
# 如果include_top = False(表示用神经网络进行特征提取), 此时需要指定输入图片尺寸. 如果include_top = True(表示神经网路被用来进行重新训练或fine-tune), 则图片输入尺寸必须在有效范围内(width & height 大于48)或和加载权重训练时的输入保持一致.
model_vgg = VGG16(weights='imagenet', include_top=False)

# .h5文件是层次数bai据格式第du5代的版本（Hierarchical Data Format，HDF5），它是用于zhi存储科学数据的一dao种文件格式zhuan和库文件。
#model_vgg = load_model('./model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', compile=False)


# 定义一种加载和预处理图像的方法
def modelProcess(img_path, model):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    # expand_dims用于扩展数组的形状，就是增加一个维度
    # np.expand_dims(a, axis=0)表示在0位置添加数据
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    # 特征提取,由于已经去掉全连接层，因此此处结果为全连接层之前特征
    x_vgg = model.predict(x)
    # 输出 (1, 7, 7, 512) = 25088
    print(x_vgg.shape)
    x_vgg = x_vgg.reshape(1, 7 * 7 * 512)
    return x_vgg


# 列出训练数据集的文件名
folder = "./icon/dataset/data_vgg/cats"
dirs = os.listdir(folder)
# 生成图像的路径
img_path = []
for i in dirs:
    if os.path.splitext(i)[1] == ".jpg":
        img_path.append(i)
img_path = [folder + "//" + i for i in img_path]

# 预处理多幅图像
features1 = np.zeros([len(img_path), 25088])
for i in range(len(img_path)):
    feature_i = modelProcess(img_path[i], model_vgg)
    print('preprocessed:', img_path[i])
    features1[i] = feature_i

folder = "./icon/dataset/data_vgg/dogs"
dirs = os.listdir(folder)
img_path = []
for i in dirs:
    if os.path.splitext(i)[1] == ".jpg":
        img_path.append(i)
img_path = [folder + "//" + i for i in img_path]
features2 = np.zeros([len(img_path), 25088])
for i in range(len(img_path)):
    feature_i = modelProcess(img_path[i], model_vgg)
    print('preprocessed:', img_path[i])
    features2[i] = feature_i

# 标签结果
print(features1.shape, features2.shape)
y1 = np.zeros(300)
y2 = np.ones(300)

# 生成训练数据
X = np.concatenate((features1, features2), axis=0)
y = np.concatenate((y1, y2), axis=0)
y = y.reshape(-1, 1)
print(X.shape, y.shape)

# 分割训练和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
print(X_train.shape, X_test.shape, X.shape)

model = Sequential()
# 25088 = 7 * 7 * 512
model.add(Dense(units=10, activation='relu', input_dim=25088))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=50)

y_train_predict = model.predict_classes(X_train)
print(accuracy_score(y_train, y_train_predict))
# 测试集准确率
y_test_predict = model.predict_classes(X_test)
print(accuracy_score(y_test, y_test_predict))

# 保存模型和加载模型
model.save('cat_dog_cnn_vgg16.h5')

# 预测自己下载的猫狗图片
a = [i for i in range(1, 10)]
fig = plt.figure(figsize=(10, 10))
for i in a:
    img_name = "./icon/" + str(i) + '.jpg'
    img_path = img_name
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    x_vgg = model_vgg.predict(x)
    x_vgg = x_vgg.reshape(1, 25088)
    result = model.predict_classes(x_vgg)
    img_ori = load_img(img_name, target_size=(250, 250))
    plt.subplot(3, 3, i)
    plt.imshow(img_ori)
    plt.title('predict：dog' if result[0][0] == 1 else 'predict：cat')
plt.show()
