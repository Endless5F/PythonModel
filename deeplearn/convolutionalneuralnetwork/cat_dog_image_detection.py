# -*-coding:utf-8-*-
# 猫狗图片检测 CNN

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from matplotlib import pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

train_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(directory="./icon/dataset/training_set", target_size=(50, 50),
                                                 batch_size=32, class_mode='binary')
# 打印分类结果标签, {'cats': 0, 'dogs': 1}
print(training_set.class_indices)

model = Sequential()
# 卷积层
model.add(Conv2D(32, (3, 3), input_shape=(50, 50, 3), activation='relu'))
# 池化层
model.add(MaxPool2D(pool_size=(2, 2)))
# 卷积层
model.add(Conv2D(32, (3, 3), activation='relu'))
# 池化层
model.add(MaxPool2D(pool_size=(2, 2)))
# 展开层 flattening layer
model.add(Flatten())
# 全连接层
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 训练模型
model.fit_generator(training_set, epochs=20)
# 训练数据准确率
print(model.evaluate_generator(training_set))
# 测试数据准确率
test_set = train_datagen.flow_from_directory(directory="./icon/dataset/test_set", target_size=(50, 50),
                                             batch_size=32, class_mode='binary')
print(model.evaluate_generator(test_set))

# 保存模型和加载模型
model.save('cat_dog_cnn.keras')

# 预测自己下载的猫狗图片
a = [i for i in range(1, 10)]
fig = plt.figure(figsize=(10, 10))
for i in a:
    # 图片地址
    img_name = "./icon/" + str(i) + '.jpg'
    # 加载图片
    img_ori = load_img(img_name, target_size=(50, 50))
    # 单个图片转化为图片数组
    img = img_to_array(img_ori)
    img = img.astype('float32') / 255
    # 得到一个四维数组, 一维一列，二维50列，三维30列，四维为rgb三列
    img = img.reshape(1, 50, 50, 3)
    # 预测
    result = model.predict_classes(img)
    img_ori = load_img(img_name, target_size=(250, 250))
    plt.subplot(3, 3, i)
    plt.imshow(img_ori)
    plt.title('predict：dog' if result[0][0] == 1 else 'predict：cat')
plt.show()
