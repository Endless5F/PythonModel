# PythonModel

## jupyter notebook

1. 安装jupyter notebook：Terminal -> pip install jupyter notebook
2. 运行jupyter notebook：Terminal -> jupyter notebook

## 项目目录简介

1. deeplearn：深度学习
    * multilayerperception：多层感知
    * convolutionalneuralnetwork：卷积神经网络
    * recurrentneuralnetwork：循环神经网络
2. machinelearn：机器学习
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
3. modelevaluationoptimization：模型评价优化
4. migrationhybridmodel：迁移混合模型
