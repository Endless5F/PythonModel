# -*-coding:utf-8-*-
# 任务：基于flare文本数据，建立LSTM模型，预测序列文字：
# 1.完成数据预处理，将文字序列数据转化为可用于LSTM输入的数据
# 2.查看文字数据预处理后的数据结构，并进行数据分离操作
# 3.针对字符串输入(” flare is a teacher in ai industry. He obtained his phd in Australia.”)，预测其对应的后续字符
# 备注：模型结构：单层LSTM，输出有20个神经元；每次使用前20个字符预测第21个字符

import utils
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import accuracy_score

data = open('./csv/flare').read()
data = data.replace('\n', '').replace('\r', '')
# 字符去重处理
letters = list(set(data))
print(letters)
num_letters = len(letters)
# 建立字典
# enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
int_to_char = {a: b for a, b in enumerate(letters)}
print(int_to_char)
char_to_int = {b: a for a, b in enumerate(letters)}
print(char_to_int)

time_step = 20

# 从文本数据中提取X和y
X, y = utils.data_preprocessing(data, time_step, num_letters, char_to_int)
print(y)
# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)
# to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示
y_train_category = to_categorical(y_train, num_letters)
# print(y_train_category)

model = Sequential()
model.add(LSTM(units=20, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
model.add(Dense(units=num_letters, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train_category, batch_size=500, epochs=5)

y_train_predict = model.predict_classes(X_train)
print(y_train_predict)

y_train_predict_char = [int_to_char[i] for i in y_train_predict]
print(y_train_predict_char)
print(accuracy_score(y_train, y_train_predict))

y_test_predict = model.predict_classes(X_test)
print(y_test_predict)
print(accuracy_score(y_test, y_test_predict))

new_letters = 'flare is a teacher in ai industry. He obtained his phd in Australia.'
X_new, y_new = utils.data_preprocessing(new_letters, time_step, num_letters, char_to_int)
y_new_predict = model.predict_classes(X_new)
y_new_predict_char = [int_to_char[i] for i in y_new_predict]
print(y_new_predict_char)

# 直观的通过log查看 输入的前20个字符，以及输出的预测字符
for i in range(0, X_new.shape[0] - 20):
    print(new_letters[i:i + 20], '--predict next letter is---', y_new_predict_char[i])
