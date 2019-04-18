# 深度学习中Loss理解
## 简介

1. 分类问题：目标变量是离散的，最终节点的输出结果（wx+b）为某个目标属于每个类别的置信度（一个向量，维度为类别个数，可以理解channel数），需要将置信度归一化到（0,1）区间内的概率分布，后使用交叉熵损失函数进行网络参数优化( Cross Entropy Loss)；

2. 回归问题：目标变量是连续的，这类问题需要预测的不是一个事先定义好的类别，而是一个任意实数，最终节点的输出结果（wx+b）就是预测值，常用的损失函数是均方误差( MSE，mean squared error )、平均绝对误差（Mean Absolute Error），就是我们常说的L2 loss、L1 loss、Smooth L1 loss。

## 提出的几点问题
1. 熵、交叉熵的含义
2. Cross Entropy Loss的使用流程
3. 为什么不能使用MSE代替Cross Entropy Loss作为分类问题的Loss
4. tensorflow中softmax_cross_entropy_with_logits与sparse_softmax_cross_entropy_with_logits函数的使用区别
5. L2 loss、L1 loss的优缺点，为什么回归问题经常使用Smooth L1 loss来代替L1 loss
6. deep learning在实战中还有那些常用的Loss

#### 1. 熵、交叉熵的含义
&emsp;&emsp;**信息量**：信息量用来衡量一个事件的不确定性，通俗讲：一个事发生概率越小，当它发生时，所携带的信息量就越大，该事件的不确定性就越大。（小菜班里倒数第一，一次考试考了全班第一，这个事件所携带的信息量就很大，所以大家都说“这谁能想到啊？”，代表的意思就是：我们" 不确定这事会发生" 的概率很大。）
&emsp;&emsp;因此信息量可用如下形式表示： 
	$$I(x) = -log(p(x))$$
	$p(x)表示事件x发生的概率，I(x)表示x发生时所携带的信息量，概率越小，发生时的信息量就越大。$
	
&emsp;&emsp;**熵**：熵是用来衡量一个系统的不确定性，代表一个系统中信息量的总和；信息量总和越大，表明这个系统不确定性就越大，系统就越不稳定。数学表示为：对所有可能事件发生时的信息量求期望，就是熵。
&emsp;&emsp;假设将某个事件$x$发生与不发生的情况看做一个系统，即满足0-1分布($x$取1或0)。假设$p(x)代表发生的概率$，则当$p(x)=0.5$时，不确定度最大（此时不发生的概率也为0.5，没有任何先验知识）；当$p(x)=0$或1时，熵为0，即此时x的取值完全确定。该系统的熵为：
$$H(x)=-[p(x)*log(p(x)+(1-p(x))*log(1-p(x))]$$
&emsp;&emsp;对于一个随机变量$X$，它所有可能取值$x\in X$的信息量的期望就为熵。
$$H(X)=-\sum p(x)logp(x)$$

&emsp;&emsp;**相对熵**：在说交叉熵之前，先说一下相对熵。相对熵又称为：KL散度（Kullback-Leibler divergence)，KL距离。它表示：预测分布$q$的信息量与真实分布$p$的信息量的差距，说白了，也就是描述预测概率$q$与真实概率$p$的差距。在分类问题中，可以理解为：当某个目标的ground truth在各类别上的概率分布为$p$时，预测结果在各类别上的概率分布为$q$，让二者之间的分布差异最小，例如某个目标属于3个类别的gt概率分布为：（1.0，0 ， 0），预测概率分布为（0.7,  0.2，0,1），则相对熵描述的就是这两个概率分布的差距。公式描述如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190410194639381.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzNDQwMDQ5,size_16,color_FFFFFF,t_70)
为了保证连续性，做如下约定：$0\log0/0 = 0，0log0/q = 0，plogp/0=\infty$。显然，当$q=p$时，两者的相对熵最小（KL距离最小）。**综上所述，相对熵的意义很明确了：表示在真实分布为$p$的前提下，使用预测概率$q$ 的信息量进行熵编码（求熵运算），与使用真实概率$p$的信息量进行熵编码的差距。在分类任务上，也就是某个目标类别的预测概率$q$与真实概率$p$的差距。**
		
						

#### 2. Cross Entropy Loss的使用流程
#### 3. 为什么不能使用MSE代替Cross Entropy Loss作为分类问题的Loss
#### 4. tensorflow中softmax_cross_entropy_with_logits、sparse_softmax_cross_entropy_with_logits函数的使用区别
#### 5. L2 loss、L1 loss的优缺点，为什么回归问题经常使用Smooth L1 loss来代替L1 loss
#### 6. deep learning在实战中还有那些常用的Loss
