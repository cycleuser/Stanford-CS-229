# CS229 课程讲义中文翻译
CS229 Lecture notes

|原作者|翻译|
|---|---|
| Raphael John Lamarre Townshend|[CycleUser](https://www.zhihu.com/people/cycleuser/columns)|


|相关链接|
|---|
|[Github 地址](https://github.com/Kivy-CN/Stanford-CS-229-CN)|
|[知乎专栏](https://zhuanlan.zhihu.com/MachineLearn)|
|[斯坦福大学 CS229 课程网站](http://cs229.stanford.edu/)|
|[网易公开课中文字幕视频](http://open.163.com/movie/2008/1/M/C/M6SGF6VB4_M6SGHFBMC.html)|

# 决策树(Decision Trees)

接下来就要讲决策树了,这是一类很简单但很灵活的算法.首先要考虑决策树所具有的非线性/基于区域(region-based)的本质,然后要定义和对比基于区域算则的损失函数,最后总结一下这类方法的具体优势和不足.讲完了这些基本内容之后,接下来再讲解通过决策树而实现的各种集成学习方法,这些技术很适合这些场景.

## 1 非线性(Non-linearity)

决策树是我们要讲到的第一种内在非线性的机器学习技术(inherently non-linear machine
learning techniques),与之形成对比的就是支持向量机(SVMs)和通用线性模型(GLMs)这些方法.严格来说,如果一个方法满足下面的特性就称之为线性方法:对于一个输入$x\in R^n$(截距项 intercept term $x_0=1$),只生成如下形式的假设函数(hypothesis functions)h:

$$
h(x)=\theta^Tx
$$

其中的$\theta\in R^n$.不能简化成上面的形式的假设函数(hypothesis functions)就都是非线性的(non-linear),如果一个方法产生的是非线性假设函数,那么这个方法也就是非线性的.之前我们已经看到过,对一个线性方法核化(kernelization)就是一种通过特征映射$\phi(x)$来实现非线性假设函数的方法.

决策树算法则可以直接产生非线性假设函数,不用去先生成合适的特征映射.接下来这个例子很振奋人心(加拿大风格的),假设要构建一个分类器,给定一个时间和一个地点,要来预测附近能不能滑雪.为了简单起见,时间就用一年中的月份来表示,而地点就用纬度(latitude)来表示(北纬南纬,-90°表示南极,0°表示迟到,90°表示南极).

![](https://raw.githubusercontent.com/Kivy-CN/Stanford-CS-229-CN/master/img/cs229notedtf1.png)

有代表性的数据结构如上图左侧所示.不能划分一个简单的线性便捷来正确区分数据集.不过可以很明显可以发现数据集中的空间中有可以鼓励出来的不同区域,一种划分方式如上图中右侧所示.上面这个分区过程(partitioning)开通过对输入空间$x$分割成不相连接的自己(或者区域)$R_i$:

$$
\begin{aligned}
X&= \bigcup ^n_{i=0} R_i\\
s.t. R_i \bigcap R_j&= \not{0} \quad\text{for}\quad i \ne j
\end{aligned}
$$

其中$n\in Z^+$.

## 2 区域选择(Selecting Regions)

通常来说,选择最有区域是很难的(intractable).决策树会生通过贪心法,从头到尾,递归分区(greedy, top-down, recursive partitioning)成一种近似的解决方案.这个方法是从头到尾(top-down)是因为是从原始输入空间$X$开始,先利用单一特征为阈值切分成两个子区域.然后对两个子区域选择一个再利用一个新阈值来进行分区.然后以递归模式来持续对模型的训练,总是选择一个叶节点(leaf node),一个特征(feature),以及一个阈值(threshold)来生成新的分割(split).严格来说,给定一个父区域$R_p$,一个特征索引$j$以及一个阈值$t\in R$,就可以得到两个子区域$R_1,R_2$,如下所示:

$$
\begin{aligned}
R_1 &= \{ X|X_j<t,X\in R_p\}\\
R_2 &= \{ X|X_j< \ge t,X\in R_p\}\\
\end{aligned}
$$

下面就着滑雪数据集来应用上面这样的过程.在步骤a当中,将输入空间$X$根据地理位置特征切分,阈值设置的是15,然后得到了子区域$R_1,R_2$.在步骤b,选择一个子区域(例子中选的是$R_2$)来递归进行同样的操作,选择时间特征,阈值设为3,然后生成了二级子区域$R_{21},R_{22}.在步骤c,对剩下的叶节点($R_1,R_{21},R_{22}$)任选一饿,然后继续上面的过程,知道遇到了一个给定的停止条件(stop criterion,这个稍后再介绍),然后再预测每个节点上的主要类别.

![](https://raw.githubusercontent.com/Kivy-CN/Stanford-CS-229-CN/master/img/cs229notedtf2.png)

## 3 定义损失函数(Defining a Loss Function)

这时候很自然的一个问题就是怎么去选择分割.首先要定义损失函数$L$,这个函数是在区域$R$上的一个集合函数(set function).对一个父区域切分成两个子区域$R_1,R_2$之后,可以计算福区域的损失函数$L(R_p)$,也可以计算子区域的基数加权(cardinality-weighted)损失函数$\frac{|R_1|L(R_1)+|R_2|L(R_2)}{|R_1|+|R_2|}$.在贪心分区框架(greedy partitioning framework)下,我们想要选择能够最大化损失函数减少量(decrease)的叶区域/特征和阈值:

$$
L(R_p)-\frac{|R_1|L(R_1)+|R_2|L(R_2)}{|R_1|+|R_2|}
$$

对一个分类问题,我们感兴趣的是误分类损失函数(misclassification loss)$L_{misclass}$.对于一个区域$R$,设$\hat p_c$是归于类别$c$的R中样本的分区.在R上的误分类损失函数可以写为:

$$
L_{misclass}(R)=1-\max_c(\hat p_c)
$$

这个可以理解为在预测区域$R$上的主要类别中发生错误分类的样本个数.虽然误分类损失函数是我们关心的最终值,但这个指标的对类别概率的变化并不敏感.举个例子,如下图所示的二值化分类.我们明确描述了父区域$R_p$,也描述了每个区域上的正负值的个数.

![](https://raw.githubusercontent.com/Kivy-CN/Stanford-CS-229-CN/master/img/cs229notedtf3.png)

第一个切分就是讲搜有的正值分割开来,但要注意到:

$$
L(R_p)=\frac{|R_1|L(R_1)+|R_2|L(R_2)}{|R_1|+|R_2|}=\frac{|R_1'|L(R_1')+|R_2'|L(R_2')}{|R_1'|+|R_2'|}=100
$$

这样不仅能使两个切分的损失函数相同,还能使得任意一个分割都不会降低在父区域上的损失函数.(Thus, not only can we not only are the losses of the two splits identical, but neither of the splits decrease the loss over that of the parent.)

因此我们有兴趣去定义一个更敏感的损失函数.前面已经给出过一些损失函数了,这里要用到的是交叉熵(cross-entropy)$L_{cross}$:

$$
L_{cross}(R)=-\sum_c \hat p_c \log_2 \hat p_c
$$

如果$\hat p_c=0$,则$\hat p_c \log_2 \hat p_c=0$.从信息论角度来看,交叉熵要衡量的是给定已知分布的情况下要确定输出所需要的位数(number of bits).更进一步说,从父区域到子区域的损失函数降低也就是信息获得(information gain).

要理解交叉熵损失函数比误分类损失函数相对更敏感,我们来看一下对同样一个二值化分类案例这两种损失函数的投图.从这些案例中,我们能够对损失函数进行简化,使之仅依赖一个区域$R_i$中正值分区的样本$\hat p_i$:

$$
\begin{aligned}
L_{misclass}(R)= L_{misclass}(\hat p)=1-\max(\hat p,1-\hat p)\\

L_{cross}(R)=L_{cross}(\hat p)=- \hat p\log \hat p -(1-\hat p)\log(1-\hat p)\\
\end{aligned}
$$

![](https://raw.githubusercontent.com/Kivy-CN/Stanford-CS-229-CN/master/img/cs229notedtf4.png)



## 4 其他考虑(Other Considerations)

### 4.1 分类变量(Categorical Variables)

### 4.2 规范化(Regularization)

### 4.3 运行(Runtime)

### 4.4 加性结构缺失(Lack of Additive Structure)

## 5 本章概要