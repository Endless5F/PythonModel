# PythonModel

# 项目依赖包

1. 导出所有项目的依赖包命令：pip3 freeze > requirements.txt
2. 安装项目依赖的时候使用命令：pip3 install -r requirements.txt

## jupyter notebook

1. 安装jupyter notebook：Terminal -> pip3 install jupyter notebook
2. 运行jupyter notebook：Terminal -> jupyter notebook

## 项目目录简介

0. 术语
    * 去均值：把输入数据各个维度都中心化为0，如下图所示，其目的就是把样本的中心拉回到坐标系原点上。
    * 归一化：幅度归一化到同样的范围，如下所示，即减少各维度数据取值范围的差异而带来的干扰。
      比如，我们有两个维度的特征A和B，A范围是0到10，而B范围是0到10000，如果直接使用这两个特征是有问题的，好的做法就是归一化，即A和B的数据都变为0到1的范围。
    * PCA/白化：用PCA降维；白化是对数据各个特征轴上的幅度归一化
1. deeplearn：深度学习
    * Keras 中文文档：https://keras.io/zh/
    * 安装TensorFlow：pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple/
    * multilayerperception：多层感知
        * 向量修改模板：f(w ⃗ * X0 ⃗) ， ⃗ 代表向量表达式的箭头
        * 人工神经元的基本组成：输入数据、激励值计算、激活函数、Output
        * 激励值计算：A = ∑i=1-n (w[i] * x[i] + b)，写成向量形式为：Output = f(w⃗ * x⃗)。
          []代表角标，x[i]小写x代表输入X0(所有第一层输入数据)第i节点数据 A为下一层节点的边界函数，w[i]为各个input节点到下一层节点的权重，b为对应偏置值
        * 激活函数：Sigmoid函数(值域(0,1))、Tanh函数(f(x)= (e^x + e^(−x)) / (e^x − e^(−x))，值域(-1,1))、ReLU函数(线性修正单元激活函数)
        * 前向传播：第一层为输入层，输入输出都为X0(向量，或者说输入矩阵)。第二层为隐含层，输入X0，输出 X1 = f(w1⃗ * X0⃗)。第三层为输出层，输入X1，输出f(w2⃗ * X1⃗)
        * 反向传播：更新模型中的权重，我们需要从最后一层输出层开始，一层一层向前计算梯度、更新参数。我们需要求出损失函数对它们三者的偏导数，找到最优的w1 w2 w3使得误差最小。
        * 梯度消失和梯度爆炸(主要取决于激活函数的选择)：
            1. 梯度消失问题：当梯度较小时，可能会产生梯度消失问题。以sigmoid函数为例，sigmoid函数的导数最大值约为0.25。考虑一个5层的网络，传递到第一层时，梯度将会衰减为(0.25)^5 =
               0.0009，当小数过小超出表示精度时，计算机会将它作为机器零来处理，因此就会导致初始几层的参数基本不会更新。
            2. 梯度爆炸问题：当梯度较大时，可能会产生梯度爆炸问题。当输出层梯度大于1时，经过多层传递，很可能导致前几层的梯度非常巨大，每一次训练参数变化很大，使得模型训练困难，也很容易“走”出合理的区域。
            3. 解决方案是改进激活函数：
                1. 一种改进的激活函数就是前面介绍的ReLU函数。它的一大特点是未激活时梯度为0，激活后梯度恒为1，由于0和1在指数运算时的不变性，就可以有效地防止梯度消失和梯度爆炸问题。
                2. 梯度裁剪。简要地说就是设定一个阈值，如果求出来的梯度大于这个阈值，我们就将梯度强行缩减为等于阈值。这样也可以防止梯度爆炸问题。
        * 选择隐藏层数量和大小的标准：
            1. 基于输入层和输出层大小：关于隐藏层大小的经验法则是在输入层和输出层之间，为了计算隐藏层大小我们使用一个一般法则：（输入大小+输出大小）*2/3
            2. 基于关键部分：通常，我们指定尽可能多的隐藏节点作为维度[主成分]，以捕获输入数据集70-90%的方差。
            3. 实际上，我是这样做的：
                * 输入层：我的数据vactor的大小（模型中特征的数量）+1表示偏差节点，当然不包括响应变量
                * 输出层：由我的模型确定：回归（一个节点）与分类（节点数等于类数，假设softmax）
                * 隐藏层：首先，一个隐藏层的节点数等于输入层的大小。“理想”的大小更可能更小（即，输入层和输出层之间的节点数），而不是更大——同样，这只是一个经验观察，大部分观察是我自己的经验。
                  如果项目证明所需的额外时间是合理的，那么我从一个由少量节点组成的隐藏层开始，然后（正如我在上面解释的那样）向隐藏层添加节点，一次添加一个节点，同时计算泛化误差、训练误差、偏差和方差。当泛化误差下降，并在它再次开始增加之前，我可以选择该点的节点数。
    * convolutionalneuralnetwork：卷积神经网络
        * 特征检测器：卷积运算结果矩阵
        * 为什么使用卷积？和只用全连接层相比，卷积层的两个主要优势在于参数共享和稀疏连接，神经网络可以通过这两种机制减少参数，以便我们用更小的训练集来训练它，从而预防过度拟合。
            * 参数共享：特征检测如垂直边缘检测如果适用于图片的某个区域，那么它也可能适用于图片的其他区域。
              也就是说，如果你用一个3×3的过滤器检测垂直边缘，那么图片的左上角区域，以及旁边的各个区域（左边矩阵中蓝色方框标记的部分）都可以使用这个3×3的过滤器。
              每个特征检测器以及输出都可以在输入图片的不同区域中使用同样的参数，以便提取垂直边缘或其它特征。 它不仅适用于边缘特征这样的低阶特征，同样适用于高阶特征，例如提取脸上的眼睛，猫或者其他特征对象。
            * 稀疏连接：特征检测器矩阵中的0是通过卷积核(过滤器，比如：3×3)
              的卷积计算得到的，它只依赖于这个卷积核和当前3×3窗口的输入矩阵，特征检测器中的一个元素0仅与输入特征中9个相连接。而且矩阵中其它值都不会对输出的该0产生任何影响，这就是稀疏连接的概念。
        * 卷积神经网络的层级结构
            * 数据输入层/ Input layer：该层要做的处理主要是对原始图像数据进行预处理，其中包括：去均值、归一化、PCA/白化
            * 卷积计算层/ CONV layer：
                * 两个关键操作
                    * 局部关联。每个神经元看做一个滤波器(filter)
                    * 窗口(receptive field)滑动，filter对局部数据计算
                * 卷积层遇到的几个名词
                    * 步长/stride：窗口一次滑动的长度
                    * 深度depth(有时也叫通道channel)：有多少神经元(即多少个卷积核或者少过滤器，用于取多个特征)深度(通道)就是几。RGB三通道，在经过6个过滤器卷积后，通道为6
                    * 填充值/zero-padding：滑动窗口没法滑完，在原先矩阵加差的n层填充值0
                * 卷积运算：滤波器矩阵(或卷积核)和输入矩阵数据，通过滑动窗口的思想，分别对当前窗口各个矩阵位置数值乘积后想加，
                  成为当前窗口的卷积值，当滑动窗口移动完成一行时卷积运算卷积运算矩阵完成一行，知道窗口滑动结束，卷积运算特征矩阵已形成。
            * ReLU激励层 / ReLU layer：
                * 激励层的实践经验： ① 不要用sigmoid！不要用sigmoid！不要用sigmoid！ ② 首先试RELU，因为快，但要小心点 ③ 如果2失效，请用Leaky ReLU或者Maxout ④
                  某些情况下tanh倒是有不错的结果，但是很少
            * 池化层 / Pooling layer：
                * 池化层夹在连续的卷积层中间， 用于压缩数据和参数的量，减小过拟合。简而言之，如果输入是图像的话，那么池化层的最主要作用就是压缩图像。
                * 池化层的具体作用：
                    1. 特征不变性，也就是我们在图像处理中经常提到的特征的尺度不变性，池化操作就是图像的resize，
                       平时一张狗的图像被缩小了一倍我们还能认出这是一张狗的照片，这说明这张图像中仍保留着狗最重要的特征，我们一看就能判断图像中画的是一只狗，图像压缩时去掉的信息只是一些无关紧要的信息，而留下的信息则是具有尺度不变性的特征，是最能表达图像的特征。
                    2. 特征降维，我们知道一幅图像含有的信息是很大的，特征也很多，但是有些信息对于我们做图像任务时没有太多用途或者有重复，我们可以把这类冗余信息去除，把最重要的特征抽取出来，这也是池化操作的一大作用。
                    3. 在一定程度上防止过拟合，更方便优化。
                * 池化层用的方法：Max pooling 和 average pooling，而实际用的较多的是Max pooling。
            * 全连接层 / FC layer：两层之间所有神经元都有权重连接，通常全连接层在卷积神经网络尾部。也就是跟传统的神经网络神经元的连接方式是一样的
        * 卷积神经网络之优缺点
            * 优点：共享卷积核，对高维数据处理无压力、无需手动选取特征，训练好权重，即得特征分类效果好
            * 缺点：需要调参，需要大样本量，训练最好要GPU、物理含义不明确（也就说，我们并不知道没个卷积层到底提取到的是什么特征，而且神经网络本身就是一种难以解释的“黑箱模型”）
    * recurrentneuralnetwork：循环神经网络
        * 循环神经网络(Rerrent Neural Network, RNN)
          ：RNN对具有序列特性的数据非常有效，它能挖掘数据中的时序信息以及语义信息，利用了RNN的这种能力，使深度学习模型在解决语音识别、语言模型、机器翻译以及时序分析等NLP(自然语言处理)领域的问题时有所突破。
        * 损失函数(loss function)交叉熵：用于多分类的损失函数，熵越大模型越不确定，熵越小模型越确定，即优化模型目的是最小化交叉熵。 E(p,q)=−∑i=1~n(p(xi)log(q(xi)))
          p(xi) 是一个xi 关于所有数据X的概率分布函数。在机器学习中，P往往用来表示样本的真实分布，比如[1,0,0]表示当前样本属于第一类。Q用来表示模型所预测的分布，比如[0.7,0.2,0.1]
          直观的理解就是如果用P来描述样本，那么就非常完美。而用Q来描述样本，虽然可以大致描述，但是不是那么的完美，信息量不足，需要额外的一些“信息增量”才能达到和P一样完美的描述。如果我们的Q通过反复训练，也能完美的描述样本，那么就不再需要额外的“信息增量”，Q等价于P。
        * RNN前向传导：s[t] = tanh(U * x[t] + W * s[t−1])、o[t] = softmax(V * s[t])， st为t时刻隐层的状态值，为向量。 o[t]
          为t时刻输出的值（这里是输入一个x[t]就有一个输出o[t]，这个是不必要的，也可以在全部x输入完之后开始输出，根据具体应用来设计模型）
        * 反向传播目的就是求预测误差E关于所有参数(U,V,W)的梯度，即∂E/∂U、∂E/∂V和∂E/∂W。
        * 反向传播算法：
            * 其中𝑉的梯度计算是比较简单的：∂E / ∂V = Σi=1~n (((∂E(𝑡) / ∂s[t]) * s[s[t]_]`) * s[t])
                1. o[t]_ = V * s[t]
                2. ∂E(𝑡) / ∂o[t]_ = (∂E(𝑡) / ∂o[t]) * (∂o[t] / ∂o[t]_) = (∂E(𝑡) / ∂o[t]) * o[o[t]_]`，o[t]` 代表是o[t]的导数
                   ∂o[t] / ∂o[t]_
                   ==》转变成o[t]`的原因：∂E(t)/∂V 实际上是对E(t)表达式中求取V的导数(或偏导数)，因此∂o[t] / ∂o[t]_ 就是o[t]表达式对o[t]_进行求导，即∂o[t] / ∂o[t]_ == o[o[t]_]`
                3. 同理：∂E(𝑡) / ∂𝑉 = (∂E(𝑡) / ∂(𝑉*s[t])) * (∂(𝑉*s[t]) / ∂𝑉) = (∂E(𝑡) / ∂o[t]_) * s[t] = ((∂E(𝑡) /
                   ∂o[t]) * o[o[t]_]`) * s[t]，∂(𝑉*s[t]) / ∂𝑉 == s[t]
            * W和U的梯度计算： 参考 https://zhuanlan.zhihu.com/p/26892413
        * RNN 其常见架构：
            * 原始 RNN：CNN 这种网络架构的特点之一就是网络的状态仅依赖于输入，而 RNN 的状态不仅依赖于输入，且与网络上一时刻的状态有关。因此，经常用于处理序列相关的问题。
            * RNN 与 BPTT：RNN 的训练跟 CNN、DNN 本质一样，依然是 BP。但它的 BP 方法名字比较高级，叫做 BPTT（Back Propagation Through Time）。
            * BRNN（Bi-directional RNN）双向循环神经网络：是单向 RNN 的一种扩展形式。普通 RNN 只关注上文，而 BRNN 则同时关注上下文，能够利用更多的信息进行预测。
              结构上BRNN由两个方向相反的 RNN 构成，这两个 RNN 连接着同一个输出层。这就达到了上述的同时关注上下文的目的。
            * DRNN(深层循环神经网络)：DRNN可以增强模型的表达能力，主要是将每个时刻上的循环体重复多次，每一层循环体中参数是共享的，但不同层之间的参数可以不同。
            * LSTM(长短期存储记忆网络)：LSTM是为了避免长依赖问题而精心设计的。记住较长的历史信息实际上是他们的默认行为，而不是他们努力学习的东西。
            * GRU(门控循环单元)
              ：GRU是LSTM网络的一种效果很好的变体，它较LSTM网络的结构更加简单，而且效果也很好，因此也是当前非常流形的一种网络。GRU既然是LSTM的变体，因此也是可以解决RNN网络中的长依赖问题。

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
            * 信息熵(Entropy)：H(X)= −Σi=1~n (p(x[i]) * log2p(x[i]))
              X代表样本总量 x[i]代表X中某一类别 p(x[i])代表随机事件X为 x[i]的概率
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
    * 迁移模型：https://tutorial.transferlearning.xyz/
    * 混合模型：监督+无监督、机器学习+深度学习
        * 数据决定模型表现上限 期望：更多高质量数据，正常数据、穷尽类别、标注正确 现实：大部分普通数据，夹杂异常数据、只包含部分类别、标注标准不一致
        * 半监督学习：是监督学习和无监督学习想结婚的一种学习方法，它同时利用有标记样本和无标记样本进行学习
        * 半监督学习目的：在标记样本有限的情况下，尽可能识别出总样本的共同特性
        * 半监督学习不局限于某种特定的方式，实现监督+无监督的灵活运用！
        * 有标签数据提取特征的半监督学习：
            1. 用有标签数据训练网络
            2. 通过隐藏层提取特征，基于特征数据对无标签数据进行建模预测
        * 思考：有标签数据是否仅限于自己收集导的数据？ 答案：可以利用别人使用的类似数据基于建模，比如：利用VGG16特区图像特征
        * 遇到复杂的聚类问题怎么办？1. 传统聚类算法效果不好2. MLP无法直接实现聚类 答案：机器学习+深度学习，比如：利用深度学习提取特征，并通过PCA降维数据，最后通过聚类算法预测数据
5. 其它
    * 熵：所有可能结果的信息量的总和组成熵。熵表征的是期望的稳定性，值越小越稳定，当熵为0时该事件为必然事件，熵越大表示该事件的可能性越难以估量。
    * 信息熵由来：
        0. 信息量：信息量是对信息的度量，就跟时间的度量是秒一样。越小概率的事情发生了产生的信息量越大，如xxx产生地震了；越大概率的事情发生了产生的信息量越小，如太阳从东边升起来了（肯定发生嘛，没什么信息量）。
        1. 信息熵公式：H(X)= −Σi=1~n (p(x[i]) * log2p(x[i]))，p(x[i])代表随机事件X为 x[i]的概率
        3. 如果我们有俩个不相关的事件x和y，那么我们观察到的俩个事件同时发生时获得的信息量应该等于观察到的事件各自发生时获得的信息之和，即：h(x,y) = h(x) + h(y)
        3. 由于x，y是俩个不相关的事件，那么满足x，y同时发生的概率：p(x,y) = p(x) * p(y)。
        4. h(x)一定与p(x)的对数有关(因为只有对数形式的真数相乘之后，能够对应对数的相加形式，log(X * Y) = logX+ logY)。 因此我们有信息量公式如下：h(x)=-log_{2}p(x)
        5. 信息量是一个具体事件发生了所带来的信息，而熵则是在结果出来之前对可能产生的信息量的期望——考虑该随机变量的所有可能取值. 即所有可能发生事件所带来的信息量的期望， 即H(x)=-sum(p(x)log_{b}p(x))
           ，b一般为2或者e(即一般是log以2为底或者以e为底 p(x)的对数)
    * 相对熵(KL散度)：Dkl(p|q)= −Σi=1~n (p(x[i]) * log2(p(x[i]) / q(x[i])))
        * 其中p(x)，q(x)为同一随机变量两个独立的概率分布。举个栗子，在机器学习的问题中常用p(x)表示label，q(x)表示predict值，如p(x)=[1，0，0]表示该样本属于第一类，q(x)
          =[0.9，0.3，0.2]表示预测该样本有0.9的概率属于第一类，显然这里用p(x)来描述样本是十分完美的，而用q(x)描述样本只能得到一个大致的描述，不是那么完美，缺少了一些信息量。经过反复的训练最后q(x)
          =[1，0，0]即可以完美描述该样本则，DKL=0。
        * 在一定程度上面，相对熵可以理解为两个随机变量之间的距离。当两个随机分布相同的时候，他们的相对熵为0，当两个随机分布的差别增大的时候，他们之间的相对熵也会增大。 但这样的理解是不够准确的，因为相对熵是不具有对称性的
        * 在机器学习中，我们常将相对熵（交叉熵是相对熵的一部分）定义为损失函数loss，我们train的目标就是要minimize(loss),其中为p(x)真实的标签，q(x)为预测值。基于以上将相对熵定义为，q(x)刻画p(
          x)的难易程度，即为其值越小则q(x)刻画p(x)越简单，当相对熵为0时，预测值完美刻画真实样本。
    * 交叉熵：E(p,q)=−∑i=1~n(p(xi)log(q(xi)))  交叉熵主要用于度量两个概率分布间的差异性信息，交叉熵 == 相对熵 - 信息熵。
    * 常用的概率分布：伯努利分布、二项式分布、多项式分布、先验概率，后验概率
        * 伯努利分布(bernouli distribution)：又叫做0-1分布，指一次随机试验，结果只有两种。最简单的例子就是，抛一次硬币，预测结果为正还是反。
        * 二项式分布(binomial distrubution)：表示n次伯努利实验的结果。记为：X~B(n,p)
          ，其中n表示实验次数，p表示每次伯努利实验的结果为1的概率，X表示n次实验中成功的次数。例子就是，求多次抛硬币，预测结果为正面的次数。
        * 多项式分布(multinomial distribution)：多项式分布是二项式分布的扩展，不同的是多项式分布中，每次实验有n种结果。最简单的例子就是多次抛筛子，统计各个面被掷中的次数。
        * 先验概率和后验概率：先验概率和后验概率的概念是相对的，后验的概率通常是在先验概率的基础上加入新的信息后得到的概率，所以也通常称为条件概率。
          比如抽奖活动，5个球中有2个球有奖，现在有五个人去抽，小名排在第三个，问题小明抽到奖的概率是多少？ 初始时什么都不知道，当然小明抽到奖的概率P(
          X = 1 ) = 2/5。但当知道第一个人抽到奖后，小明抽到奖的概率就要发生变化，P(X = 1| Y1 = 1) = 1/4。再比如自然语言处理中的语言模型，需要计算一个单词被语言模型产生的概率P(w)
          。没有看到任何语料库的时候，我们只能猜测或者平经验，或者根据一个文档中单词w的占比，来决定单词的先验概率P(w) =
          1/1000。之后根据获得的文档越多，我们可以不断的更新。再比如，你去抓娃娃机，没抓之前，你也可以估计抓到的概率，大致在1/5到1/50之间，它不可能是1/1000或1/2。然后你可以通过投币，多次使用娃娃机，更据经验来修正，你对娃娃机抓到娃娃的概率推断。后验概率有时候也可以认为是不断学习修正得到的更精确，或者更符合当前情况下的概率。
        * 共轭分布：通常我们可以假设先验概率符合某种规律或者分布，然后根据增加的信息，我们同样可以得到后验概率的计算公式或者分布。
          如果先验概率和后验概率的符合相同的分布，那么这种分布叫做共轭分布。共轭分布的好处是可以清晰明了的看到，新增加的信息对分布参数的影响，也即概率分布的变化规律。
    * batch_size的含义：批量梯度下降(小批量梯度下降)。深度学习的优化算法就是梯度下降，每次的参数更新有两种方式。
        * 第一种，遍历全部数据集算一次损失函数，然后算函数对各个参数的梯度，更新梯度。这种方法每更新一次参数都要把数据集里的所有样本都看一遍，计算量开销大，计算速度慢，不支持在线学习，这称为Batch gradient
          descent，批梯度下降。
        * 另一种，每看一个数据就算一下损失函数，然后求梯度更新参数，这个称为随机梯度下降，stochastic gradient
          descent。这个方法速度比较快，但是收敛性能不太好，可能在最优点附近晃来晃去，hit不到最优点。两次参数的更新也有可能互相抵消掉，造成目标函数震荡的比较剧烈。
        * 为了克服两种方法的缺点，现在一般采用的是一种折中手段，mini-batch gradient
          decent，小批的梯度下降，这种方法把数据分为若干个批，按批来更新参数，这样，一个批中的一组数据共同决定了本次梯度的方向，下降起来就不容易跑偏，减少了随机性。另一方面因为批的样本数与整个数据集相比小了很多，计算量也不是很大。




