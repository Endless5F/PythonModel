# -*-coding:utf-8-*-
# PCA实战task：
# 1、基于iris_data.csv数据，建立KNN模型实现数据分类（n_neighbors=3）
# 2、对数据进行标准化处理，选取一个维度可视化处理后的效果
# 3、进行与原数据等维度PCA，查看各主成分的方差比例
# 4、保留合适的主成分，可视化降维后的数据
# 5、基于降维后数据建立KNN模型，与原数据表现进行对比

import pandas as pd
import numpy as np

data = pd.read_csv('./csv/iris_data.csv')
X = data.drop(['target','label'],axis=1)
y = data.loc[:,'label']
