# CS229 课程讲义中文翻译
CS229 Section notes

|原作者|翻译|
|---|---|
|Zico Kolter|[XiaoDong_Wang](https://github.com/Dongzhixiao) |


|相关链接|
|---|
|[Github 地址](https://github。com/Kivy-CN/Stanford-CS-229-CN)|
|[知乎专栏](https://zhuanlan。zhihu。com/MachineLearn)|
|[斯坦福大学 CS229 课程网站](http://cs229。stanford。edu/)|
|[网易公开课中文字幕视频](http://open。163。com/movie/2008/1/M/C/M6SGF6VB4_M6SGHFBMC。html)|


### 凸优化概述

#### 1. 介绍

在很多时候，我们进行机器学习算法时希望优化某些函数的值。即，给定一个函数$f:R^n\rightarrow R$，我们想求出使函数$f(x)$最小化（或最大化）的原像$x\in R^n$。我们已经看过几个包含优化问题的机器学习算法的例子，如：最小二乘算法、逻辑回归算法和支持向量机算法，它们都可以构造出优化问题。

在一般情况下，很多案例的结果表明，想要找到一个函数的全局最优值是一项非常困难的任务。然而，对于一类特殊的优化问题——**凸优化问题，** 我们可以在很多情况下有效地找到全局最优解。在这里，有效率既有实际意义，也有理论意义：它意味着我们可以在合理的时间内解决任何现实世界的问题，它意味着理论上我们可以在一定的时间内解决该问题，而时间的多少只取决于问题的多项式大小。**（译者注：即算法的时间复杂度是多项式级别$O(n^k)$，其中$k$代表多项式中的最高次数）**

这部分笔记和随附课程的目的是对凸优化领域做一个非常简要的概述。这里的大部分材料（包括一些数字）都是基于斯蒂芬·博伊德(Stephen Boyd)和利文·范登伯格(lieven Vandenberghe)的著作《凸优化》（凸优化<a target='_blank' href='https://web.stanford.edu/~boyd/cvxbook/'>[1]</a>在网上<a target='_blank' href='https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf'>免费提供下载</a>）以及斯蒂芬·博伊德(Stephen Boyd)在斯坦福教授的课程EE364。如果您对进一步研究凸优化感兴趣，这两种方法都是很好的资源。

#### 2. 凸集

我们从**凸集**的概念开始研究凸优化问题。

**定义2.1** 我们定义一个集合是凸的，当且仅当任意$x,y\in C$ 且 $\theta\in R, 0\le\theta\le 1$,

$$
\theta x + (1-\theta)y\in C
$$

实际上，这意味着如果我们在集合$C$中取任意两个元素，在这两个元素之间画一条直线，那么这条直线上的每一点都属于$C$。图$1$显示了一个示例的一个凸和一个非凸集。其中点$\theta x +(1-\theta) y$被称作点集$x,y$的**凸性组合(convex combination)。**

![](https://raw.githubusercontent.com/Kivy-CN/Stanford-CS-229-CN/master/img/cs229notecof1.png)

##### 2.1 凸集的实例

- **全集$R^n$** 是一个凸集 **（译者注：即$n$维向量空间或线性空间中所有元素组成的集合）。** 因为以下结论显而易见：任意$x,y\in R^n$，则$\theta x +(1-\theta) y\in R^n$。**（译者注：$n$维向量空间对加法和数乘封闭，显然其线性组合也封闭）**

- **非负的象限$R^n_+$组成的集合** 是一个凸集。$R^n$中所有非负的元素组成非负象限：$R^n_+=\{x:x_i\ge 0\quad\forall i=1,\dots,n \}$。为了证明$R^n_+$是一个凸集，只要给定任意$x,y\in R^n_+$，并且$0\le\theta\le 1$，

$$
(\theta x +(1-\theta) y)_i = \theta x_i + (1 - \theta)y_i\ge 0\quad\forall i
$$

- **范数球**是一个凸集。设$\parallel\cdot\parallel$是$R^n$中的一个范数（例如，欧几里得范数，即二范数$\parallel x\parallel_2=\sqrt{\sum_{i=1}^nx_i^2}$）。则集合$\{x:\parallel x\parallel\le 1\}$是一个凸集。为了证明这个结论，假设$x,y\in R^n$，其中$\parallel x\parallel\le 1,\parallel y\parallel\le 1,0\le\theta\le 1$，则：

$$
\parallel \theta x +(1-\theta) y\parallel\le \parallel\theta x\parallel+\parallel(1-\theta) y\parallel = \theta\parallel x\parallel+(1-\theta)\parallel y\parallel\le 1
$$

我们用了三角不等式和范数的正同质性(positive homogeneity)。

- **仿射子空间和多面体** 是一个凸集。给定一个矩阵$A\in R^{m\times n}$和一个向量$b\in R^m$，一个仿射子空间可以表示成一个集合$\{x\in R^n:Ax=b\}$（注意如果$b$无法通过$A$列向量的线性组合得到时，结果可能是空集）。类似的，一个多面体（同样，也可能是空集）是这样一个集合$\{x\in R^n:Ax\preceq b\}$，其中‘$\preceq$’代表分量不等式(componentwise inequality)（也就是，$Ax$得到的向量中的所有元素都小于等于$b$向量对应位置的元素）$^1$。为了证明仿射子空间和多面体是凸集，首先考虑$x,y\in R^n$，这样可得$Ax=Ay=b$。则对于$0\le\theta\le 1$，有：

>1 类似的，对于两个向量$x,y\in R^n$，$x\succeq y$代表，向量$x$中的每一个元素都大于等于向量$y$对应位置的元素。注意有时候文中使用符号‘$\le$’和‘$\ge$’代替了符号‘$\preceq$’和‘$\succeq$’，则符号的实际意义必须根据上下文来确定（也就是如果等式两边都是向量的时候，文中使用的是常规的不等号‘$\le$’和‘$\ge$’，则我们自己心中要知道用后两个符号‘$\preceq$’和‘$\succeq$’的意义代替之）

$$
A(\theta x +(1-\theta) y) = \theta Ax + (1-\theta)Ay=\theta b + (1-\theta)b=b
$$

类似的，对于$x,y\in R^n$，满足$Ax\le b$以及$Ay\le b,0\le\theta\le 1$，则：

$$
A(\theta x +(1-\theta) y) = \theta Ax + (1-\theta)Ay\le\theta b + (1-\theta)b=b
$$

- **凸集之间的交集**还是凸集。假设$C_1,C_2,\dots,C_k$都是凸集，则它们的交集：

$$
\bigcap_{i=1}^kC_i=\{x:x\in C_i\quad\forall i=1,\dots,k\}
$$

同样也是凸集。为了证明这个结论，考虑$x,y\in \bigcap_{i=1}^k C_i$以及$0\le\theta\le 1$，则：

$$
\theta x +(1-\theta) y\in C_i\quad\forall i=1,\cdots,k
$$

因此，根据凸集的定义可得：

$$
\theta x +(1-\theta) y\in\bigcap_{i=1}^kC_i
$$

然而要注意在通常情况下，凸集之间的并集并不是一个凸集

- **半正定矩阵**是一个凸集。所有对称正半定矩阵的集合，常称为正半定锥，记作$S^n_+$，其是一个凸集（通常情况下，$S^n\subset R^{n\times n}$代表$n\times n$对称矩阵的集合）。回忆一个概念，我们说一个矩阵$A\in R^{n\times n}$是对称半正定矩阵，当且仅当该矩阵满足$A=A^T$，并且给定任意一个$n$维向量$x\in R^n$，满足$x^TAx\ge 0$。现在考虑两个对称半正定矩阵$A,B\in S^n_+$，并且有$0\le\theta\le 1$。给定任意$n$维向量$x\in R^n$，则：

$$
x^T(\theta A +(1-\theta) B)x=\theta x^TAx+(1-\theta)x^TBx\ge 0
$$

同样的逻辑可以用来证明所有正定、负定和负半定矩阵的集合也是凸集。

