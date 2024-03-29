# -*-coding:utf-8-*-

import numpy as np
from keras.utils import to_categorical


# 滑动窗口提取数据
def extract_data(data, slide):
    x = []
    y = []
    for i in range(len(data) - slide):
        x.append([a for a in data[i:i + slide]])
        y.append(data[i + slide])
    return x, y


# 字符到数字的批量转化
def char_to_int_Data(x, y, char_to_int):
    x_to_int = []
    y_to_int = []
    for i in range(len(x)):
        x_to_int.append([char_to_int[char] for char in x[i]])
        y_to_int.append([char_to_int[char] for char in y[i]])
    return x_to_int, y_to_int


# 实现输入字符文章的批量处理，输入整个字符、滑动窗口大小、转化字典
def data_preprocessing(data, slide, num_letters, char_to_int):
    char_Data = extract_data(data, slide)
    int_Data = char_to_int_Data(char_Data[0], char_Data[1], char_to_int)
    Input = int_Data[0]
    Output = list(np.array(int_Data[1]).flatten())
    Input_RESHAPED = np.array(Input).reshape(len(Input), slide)
    new = np.random.randint(0, 10, size=[Input_RESHAPED.shape[0], Input_RESHAPED.shape[1], num_letters])
    for i in range(Input_RESHAPED.shape[0]):
        for j in range(Input_RESHAPED.shape[1]):
            new[i, j, :] = to_categorical(Input_RESHAPED[i, j], num_classes=num_letters)
    return new, Output
