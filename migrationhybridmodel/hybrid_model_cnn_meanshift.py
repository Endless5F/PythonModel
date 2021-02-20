# -*-coding:utf-8-*-

# 任务：机器学习+深度学习，根据original_data样本，建立模型，对test_data的图片进行普通/其他苹果判断：
# 1.数据增强，扩充确认为普通苹果的样本数量
# 2.特征提取，使用VGG16模型提取图像特征
# 3.图片批量处理
# 4.Kmeans模型尝试普通、其他苹果聚类
# 5.基于标签数据矫正结果，并可视化
# 6.Meanshift模型提升模型表现
# 7.数据降维PCA处理，提升模型表现

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import os
from sklearn.cluster import MeanShift, estimate_bandwidth
from collections import Counter
from matplotlib import pyplot as plt

path = "./icon/original_data"
save_path = "./icon/gen_data"

datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.02, horizontal_flip=True,
                             vertical_flip=True)
gen = datagen.flow_from_directory(path, target_size=(224, 224), batch_size=2, save_to_dir=save_path, save_prefix='gen',
                                  save_format='jpg')
for i in range(100):
    gen.next()

model_vgg = VGG16(weights='imagenet', include_top=False)

folder = "./icon/train_data"
dirs = os.listdir(folder)
# 名称合并
img_path_list = []
for i in dirs:
    if os.path.splitext(i)[1] == '.jpg':
        img_path_list.append(i)
img_path_list = [folder + "//" + i for i in img_path_list]


def modelProcess(image_path, model):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    X = np.expand_dims(img, axis=0)
    X = preprocess_input(X)
    X_VGG = model.predict(X)
    X_VGG = X_VGG.reshape(1, 7 * 7 * 512)
    return X_VGG


# 图像批量处理
features_train = np.zeros([len(img_path_list), 7 * 7 * 512])
for i in range(len(img_path_list)):
    feature_i = modelProcess(img_path_list[i], model_vgg)
    print('preprocessed:', img_path_list[i])
    features_train[i] = feature_i

X = features_train

# 测试数据集
folder_test = './icon/test_data'
dirs_test = os.listdir(folder_test)
img_path_test = []
for i in dirs_test:
    if os.path.splitext(i)[1] == '.jpg':
        img_path_test.append(i)
img_path_test = [folder_test + "//" + i for i in img_path_test]

features_test = np.zeros([len(img_path_test), 7 * 7 * 512])
for i in range(len(img_path_test)):
    feature_i = modelProcess(img_path_test[i], model_vgg)
    print('preprocessed:', img_path_test[i])
    features_test[i] = feature_i
X_test = features_test

# 使用均值漂移聚类
bw = estimate_bandwidth(X, n_samples=140)
cnn_ms = MeanShift(bandwidth=bw)
cnn_ms.fit(X)
y_predict_ms = cnn_ms.predict(X)
print(Counter(y_predict_ms))  # 预测结果统计
normal_apple_id = 0
fig4 = plt.figure(figsize=(10, 40))
for i in range(45):
    for j in range(5):
        img = load_img(img_path_list[i * 5 + j])  # read the image
        plt.subplot(45, 5, i * 5 + j + 1)
        plt.title('apple' if y_predict_ms[i * 5 + j] == normal_apple_id else 'others')
        plt.imshow(img), plt.axis('off')

y_predict_ms_test = cnn_ms.predict(X_test)
print(y_predict_ms_test)
fig5 = plt.figure(figsize=(10, 10))
for i in range(3):
    for j in range(4):
        img = load_img(img_path_test[i * 4 + j])  # read the image
        plt.subplot(3, 4, i * 4 + j + 1)
        plt.title('apple' if y_predict_ms_test[i * 4 + j] == normal_apple_id else 'others')
        plt.imshow(img), plt.axis('off')
plt.show()
