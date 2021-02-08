# PythonModel

# 项目依赖包

1. 导出所有项目的依赖包命令：pip3 freeze > requirements.txt
2. 安装项目依赖的时候使用命令：pip install -r requirements.txt

## jupyter notebook

1. 安装jupyter notebook：Terminal -> pip install jupyter notebook
2. 运行jupyter notebook：Terminal -> jupyter notebook

## 项目目录简介

1. deeplearn：深度学习
    * Keras 中文文档：https://keras.io/zh/
    * 安装TensorFlow：pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple/
    * multilayerperception：多层感知
    * convolutionalneuralnetwork：卷积神经网络
    * recurrentneuralnetwork：循环神经网络
2. machinelearn：机器学习
    * sklearn 中文文档：https://www.scikitlearn.com.cn/
    * linearregression：线性回归
        * 单因子线性回归 - 一元
        * 多因子线性回归实战 - 多元
    * logisticregression：逻辑回归(Sigmoid函数: f(x)=1/(1+e^−x))
        * 一阶边界函数： 𝜃0+𝜃1𝑋1+𝜃2𝑋2=0
        * 二阶边界函数： 𝜃0+𝜃1𝑋1+𝜃2𝑋2+𝜃3𝑋21+𝜃4𝑋22+𝜃5𝑋1𝑋2=0
    * clustering：聚类
        * K-means算法：K-平均或者K-均值
        * KNN，K-NearestNeighbor：邻近算法(监督学习)
        * meanShift算法：均值漂移，固定半径r
    * otheralgorithm：其它算法
        * Decision Tree：决策树算法
            * 信息熵(Entropy)：Ent(D)= −Σk=1~n (p(k) * log2p(k))
              D代表样本总量 k代表某一类别 p(k)代表某一类别在总样本类别中的比例
            * 某一属性信息增益(Information Gain)：Gain(D, a) = Ent(D) - Σv=1~n (Dv/D * Ent(Dv))
              a代表某一属性(特征) v代表属性a划分出来的类别数量,比如说a属性影响两种类别(总3种) Dv代表类别v样本数
        * Anomaly Detection：异常检测
            * 高斯分布：𝑓(𝑥) = (1/(√2𝜋 * 𝜎)) * (𝑒^(−((𝑥−𝜇)^2)/(2*𝜎^2)))，公式中μ为平均数，σ为标准差，f(x)为正态分布函数
            * 高阶高斯分布：p(x) = Πj=1~n 𝑓(j)，“Π”累乘符号 j代表对每一个维度都计算对应的正态分布
        * PCA(Principal Component Analysis)，即主成分分析方法：数据降维算法
3. modelevaluationoptimization：模型评价优化
    * 过拟合和欠拟合
        * 过拟合问题：训练数据准确，预测数据不准确
            * 原因：1. 模型结构过于复杂(维度过高)。 2. 使用了过多属性，模型训练时包含了干扰信息。
            * 解决办法：1. 简化模型结构(使用低阶模型，比如线性模型)。2. 数据预处理，保留主成分信息(数据PCA处理)。3. 在模型训练时，增加正则化项(regularization)
        * 欠拟合问题：训练数据不准确，预测数据也不准确
            * 解决办法：欠拟合可以通过观察训练数据及时发现问题，通过优化模型结果解决
    * 数据分离和混淆矩阵
        * 数据分离：1. 把数据分成两部分：训练家、测试集。2. 使用训练集数据进行模型训练。3. 使用测试集数据进行预测，更有效的评估模型对于新数据的预测表现
        * 混淆矩阵（confusion_matrix）
            * 分类评估指标中定义的一些符号含义:
                * TP(True Positive)：将正类预测为正类数，真实为0，预测也为0
                * FN(False Negative)：将正类预测为负类数，真实为0，预测为1
                * FP(False Positive)：将负类预测为正类数， 真实为1，预测为0
                * TN(True Negative)：将负类预测为负类数，真实为1，预测也为1
            * 概念：误差矩阵，用于衡量分类算法的准确程度
            * 背景场景：使用准确率进行模型评估的局限性：模型1和模型2表现的差别 源数据：1000个数据，900个1，100个0 模型1：850个1预测正确，50个个0预测准确，准确率为90%
              模型2：预测所有的样本结构都是1的准确率90% (空正确率)
            * 背景总结：准确率可以方便的用于衡量模型的整体预测效果，但无法反应细节信息，具体表现在：1. 没有体现数据预测的实际分布情况。2. 没有体现模型错误预测的类型
            * 混淆矩阵评价指标:
                * AccuracyRate(准确率): (TP+TN)/(TP+TN+FN+FP)
                * ErrorRate(误分率): (FN+FP)/(TP+TN+FN+FP)
                * Recall(召回率，查全率,击中概率): TP/(TP+FN), 在所有GroundTruth为正样本中有多少被识别为正样本了;
                * Precision(查准率):TP/(TP+FP),在所有识别成正样本中有多少是真正的正样本；
                * TPR(TruePositive Rate): TP/(TP+FN),实际就是Recall
                * FAR(FalseAcceptance Rate)或FPR(False Positive Rate)错误接收率/误报率：FP/(FP+TN)，在所有GroundTruth为负样本中有多少被识别为正样本了;
                * FRR(FalseRejection Rate)错误拒绝率/拒真率: FN/(TP+FN)，在所有GroundTruth为正样本中有多少被识别为负样本了，它等于1-Recall
    * 模型优化
        * 问题
            1. 问题一：用什么算法？
            2. 问题二：具体算法的核心结构或者参数如何选择？比如：1. 逻辑回归边界函数用线性还是多项式？2. KNN的核心参数n_neighbors取多少合适？
            3. 问题三：模型表现不佳，怎么办？
        * 如何提高模型表现？数据质量决定模型表现的上限
            1. 数据属性的意义，是否为无关数据
            2. 不同属性数据的数量级差异性如何。比如：身高1.8m和体重70kg
            3. 是否有异常数据
            4. 采集数据的方法是否合理，采集到的数据是否有代表性
            5. 对于标签结果，要确保标签判定规则的一致性(统一标准)
        * try --》 Benefits
            1. 删除不必要的属性 --》 减少过拟合、节约运算时间
            2. 数据预处理：归一化、标准化 --》 平衡数据影响，加快训练时间
            3. 确保是否保留或过滤异常数据 --》 提高鲁棒性
            4. 尝试不同的模型，对比模型表现 --》 帮助确定更合适的模型
        * 目标：在确定模型类别后，如何让模型表现更好。三个方面：数据、模型核心参数、正则化。尝试以下方法：
            1. 遍历核心参数组合，评估对应模型表现（比如：逻辑回归边界函数考虑多项式、KNN尝试不同的n_neighbors值(值越小，模型复杂度越高)）
            2. 扩大数据样本
            3. 增加或减少数据属性
            4. 对数据进行降维处理
            5. 对模型进行正则化处理，调整正则项λ的数值
        * 训练集数据准确率，随着模型复杂而越高。测试数据集准确率，在模型过于简单或者过于复杂的情况时下降。选择合适程度的模型
4. migrationhybridmodel：迁移混合模型
