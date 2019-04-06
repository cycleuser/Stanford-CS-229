# CS229 课程讲义中文翻译
CS229 Section notes

|原作者|翻译|
|---|---|
|Chuong B. Do|[XiaoDong_Wang](https://github.com/Dongzhixiao) |


|相关链接|
|---|
|[Github 地址](https://github。com/Kivy-CN/Stanford-CS-229-CN)|
|[知乎专栏](https://zhuanlan。zhihu。com/MachineLearn)|
|[斯坦福大学 CS229 课程网站](http://cs229。stanford。edu/)|
|[网易公开课中文字幕视频](http://open。163。com/movie/2008/1/M/C/M6SGF6VB4_M6SGHFBMC。html)|


### 高斯过程

#### 介绍

我们在本课程上半部分讨论的许多经典机器学习算法都符合以下模式：给定一组从未知分布中采样的独立同分布的示例训练集：

1. 求解一个凸优化问题，以确定数据单一的“最佳拟合”模型，并
2. 使用这个估计模型对未来的测试输入点做出“最佳猜测”的预测。

在本节的笔记中，我们将讨论一种不同的学习算法，称为**贝叶斯方法。** 与经典的学习算法不同，贝叶斯算法并不试图识别数据的“最佳匹配”模型(或者类似地，对新的测试输入做出“最佳猜测”的预测)。相反，其计算模型上的后验分布（或者类似地，计算新的输出的测试数据的后验预测分布）。这些分布提供了一种有用的方法来量化模型估计中的不确定性，并利用我们对这种不确定性的知识来对新的测试点做出更可靠的预测。

我们来关注下**回归**问题，即：目标是学习从某个$n$维向量的输入空间$\mathcal{X} = R^n$到实值目标的输出空间$\mathcal{Y} = R$的映射。特别地，我们将讨论一个基于核的完全贝叶斯回归算法，称为高斯过程回归。本节的笔记中涉及的内容主要包括我们之前在课堂上讨论过的许多不同主题（即线性回归$^1$的概率解释、贝叶斯方法$^2$、核方法$^3$和多元高斯$^4$的性质）。

>1 参见“[监督学习，判别算法](https://kivy-cn.github.io/Stanford-CS-229-CN/#/Markdown/cs229-notes1)”课程讲义。
>2 参见“[正则化和模型选择](https://kivy-cn.github.io/Stanford-CS-229-CN/#/Markdown/cs229-notes5)”课程讲义。
>3 参见“[支持向量机](https://kivy-cn.github.io/Stanford-CS-229-CN/#/Markdown/cs229-notes3)”课程讲义。
>4 参见“[因子分析](https://kivy-cn.github.io/Stanford-CS-229-CN/#/Markdown/cs229-notes9)”课程讲义。

本节的笔记后续内容的组织如下。在第1小节中，我们简要回顾了多元高斯分布及其性质。在第2小节中，我们简要回顾了贝叶斯方法在概率线性回归中的应用。第3小节给出了高斯过程的中心思想，第4小节给出了完整的高斯过程回归模型。

#### 1. 多元高斯分布

我们称一个概率密度函数是一个均值为$\mu\in R^n$，协方差矩阵为$\Sigma\in S_{++}^n$的$^1$一个**多元正态分布（或高斯分布）(multivariate normal (or Gaussian) distribution)，** 其随机变量是向量值$x\in R^n$，该概率密度函数可以通过下式表达：

$$
p(x;\mu,\Sigma)=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))\qquad\qquad(1)
$$

我们可以写作$x\sim\mathcal{N}(\mu,\Sigma)$。这里，回想一下线性代数的笔记中$S_{++}^n$指的是对称正定$n\times n$矩阵$^5$的空间。

>5 实际上，在某些情况下，我们需要处理的多元高斯分布的$\Sigma$是正半定而非正定（即，$\Sigma$不满值）。在这种情况下，$\Sigma^{-1}$不存在，所以$(1)$式中给出的高斯概率密度函数的定义并不适用。例子可以参阅“[因子分析](https://kivy-cn.github.io/Stanford-CS-229-CN/#/Markdown/cs229-notes9)”课程讲义。

一般来说，高斯随机变量在机器学习和统计中非常有用，主要有两个原因。首先，它们在统计算法中建模“噪声”时非常常见。通常，噪声可以被认为是影响测量过程的大量独立的小随机扰动的累积；根据中心极限定理，独立随机变量的和趋于高斯分布。其次，高斯随机变量对于许多分析操作都很方便，因为许多涉及高斯分布的积分实际上都有简单的闭式解。在本小节的其余部分，我们将回顾多元高斯的一些有用性质。

给定随机向量$x \in R^{n}$服从多元高斯分布$x\sim\mathcal{N}(\mu,\Sigma)$。假设$x$中的变量被分成两个集合$x_{A}=\left[x_{1} \cdots x_{r}\right]^{T} \in R^{r}$和$x_{B}=\left[x_{r+1} \cdots x_{n}\right]^{T} \in R^{n-r}$（对于$\mu$和$\Sigma$也进行同样的拆分），则有：

$$
x=\left[ \begin{array}{c}{x_{A}} \\ {x_{B}}\end{array}\right] \quad \mu=\left[ \begin{array}{c}{\mu_{A}} \\ {\mu_{B}}\end{array}\right] \quad \Sigma=\left[ \begin{array}{cc}{\sum_{A A}} & {\sum_{A B}} \\ {\Sigma_{B A}} & {\Sigma_{B B}}\end{array}\right]
$$

因为$\Sigma=E\left[(x-\mu)(x-\mu)^{T}\right]=\Sigma^{T}$，所以上式中有$\Sigma_{A B}=\Sigma_{B A}^{T}$。下列性质适用：

1. **规范化。** 概率密度函数的归一化，即：

$$
\int_{x} p(x ; \mu, \Sigma) dx = 1
$$

这个特性乍一看似乎微不足道，但实际上对于计算各种积分非常有用，即使是那些看起来与概率分布完全无关的积分（参见附录A.1）！

2. **边缘化。** 边缘概率密度函数：

$$
\begin{aligned} p\left(x_{A}\right) &=\int_{x_{B}} p\left(x_{A} , x_{B} ; \mu, \Sigma\right) d x_{B} \\ p\left(x_{B}\right) &=\int_{x_{A}} p\left(x_{A}, x_{B} ; \mu,\Sigma\right) d x_{A} \end{aligned}
$$

是高斯分布：

$$
\begin{aligned} x_{A} & \sim \mathcal{N}\left(\mu_{A}, \Sigma_{A A}\right) \\ x_{B} & \sim \mathcal{N}\left(\mu_{B}, \Sigma_{B B}\right) \end{aligned}
$$

3. **条件化。** 条件概率密度函数：

$$
\begin{aligned} p\left(x_{A} | x_{B}\right) &=\frac{p\left(x_{A}, x_{B} ; \mu, \Sigma\right)}{\int_{x_{A}} p\left(x_{A}, x_{B} ; \mu, \Sigma\right) d x_{A}} \\ p\left(x_{B} | x_{A}\right) &=\frac{p\left(x_{A}, x_{B} ; \mu, \Sigma\right)}{\int_{x_{B}} p\left(x_{A}, x_{B} ; \mu, \Sigma\right) d x_{B}} \end{aligned}
$$

是高斯分布：

$$
x_{A} | x_{B} \sim \mathcal{N}\left(\mu_{A}+\Sigma_{A B} \Sigma_{B B}^{-1}\left(x_{B}-\mu_{B}\right), \Sigma_{A A}-\Sigma_{A B} \Sigma_{B B}^{-1} \Sigma_{B A}\right) \\
x_{B} | x_{A} \sim \mathcal{N}\left(\mu_{B}+\Sigma_{B A} \Sigma_{A A}^{-1}\left(x_{A}-\mu_{A}\right), \Sigma_{B B}-\Sigma_{B A} \Sigma_{A A}^{-1} \Sigma_{A B}\right)
$$

附录A.2给出了这一性质的证明。（参见附录A.3的更简单的派生版本。）

4. **求和性。** （相同维数的）独立高斯随机变量$y \sim \mathcal{N}(\mu, \Sigma)$和$z \sim \mathcal{N}\left(\mu^{\prime}, \Sigma^{\prime}\right)$之和同样是高斯分布：

$$
y+z \sim \mathcal{N}\left(\mu+\mu^{\prime}, \Sigma+\Sigma^{\prime}\right)
$$