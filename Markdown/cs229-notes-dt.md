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

## 3 定义损失函数(Defining a Loss Function)

## 4 其他考虑(Other Considerations)

### 4.1 分类变量(Categorical Variables)

### 4.2 规范化(Regularization)

### 4.3 运行(Runtime)

### 4.4 加性结构缺失(Lack of Additive Structure)

## 5 本章概要