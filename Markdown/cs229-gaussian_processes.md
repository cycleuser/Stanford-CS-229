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

在本节的笔记中，我们将讨论一种不同的学习算法，称为**贝叶斯方法。** 与经典的学习算法不同，贝叶斯算法并不试图识别数据的“最佳匹配”模型（或者类似地，对新的测试输入做出“最佳猜测”的预测）。相反，其计算模型上的后验分布（或者类似地，计算新的输出的测试数据的后验预测分布）。这些分布提供了一种有用的方法来量化模型估计中的不确定性，并利用我们对这种不确定性的知识来对新的测试点做出更可靠的预测。

我们来关注下**回归**问题，即：目标是学习从某个$n$维向量的输入空间$\mathcal{X} = R^n$到实值目标的输出空间$\mathcal{Y} = R$的映射。特别地，我们将讨论一个基于核的完全贝叶斯回归算法，称为高斯过程回归。本节的笔记中涉及的内容主要包括我们之前在课堂上讨论过的许多不同主题（即线性回归$^1$的概率解释、贝叶斯方法$^2$、核方法$^3$和多元高斯$^4$的性质）。

>1 参见“[监督学习，判别算法](https://kivy-cn.github.io/Stanford-CS-229-CN/#/Markdown/cs229-notes1)”课程讲义。

>2 参见“[正则化和模型选择](https://kivy-cn.github.io/Stanford-CS-229-CN/#/Markdown/cs229-notes5)”课程讲义。

>3 参见“[支持向量机](https://kivy-cn.github.io/Stanford-CS-229-CN/#/Markdown/cs229-notes3)”课程讲义。

>4 参见“[因子分析](https://kivy-cn.github.io/Stanford-CS-229-CN/#/Markdown/cs229-notes9)”课程讲义。

本节的笔记后续内容的组织如下。在第1小节中，我们简要回顾了多元高斯分布及其性质。在第2小节中，我们简要回顾了贝叶斯方法在概率线性回归中的应用。第3小节给出了高斯过程的中心思想，第4小节给出了完整的高斯过程回归模型。

#### 1. 多元高斯分布

我们称一个概率密度函数是一个均值为$\mu\in R^n$，协方差矩阵为$\Sigma\in S_{++}^n$的一个**多元正态分布（或高斯分布）(multivariate normal (or Gaussian) distribution)，** 其随机变量是向量值$x\in R^n$，该概率密度函数可以通过下式表达：

$$
p(x;\mu,\Sigma)=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))\qquad\qquad(1)
$$

我们可以写作$x\sim\mathcal{N}(\mu,\Sigma)$。这里，回想一下线性代数的笔记中$S_{++}^n$指的是对称正定$n\times n$矩阵$^5$的空间。

>5 实际上，在某些情况下，我们需要处理的多元高斯分布的$\Sigma$是正半定而非正定（即，$\Sigma$不满值）。在这种情况下，$\Sigma^{-1}$不存在，所以$(1)$式中给出的高斯概率密度函数的定义并不适用。例子可以参阅“[因子分析](https://kivy-cn.github.io/Stanford-CS-229-CN/#/Markdown/cs229-notes9)”课程讲义。

一般来说，高斯随机变量在机器学习和统计中非常有用，主要有两个原因。首先，它们在统计算法中建模“噪声”时非常常见。通常，噪声可以被认为是影响测量过程的大量独立的小随机扰动的累积；根据中心极限定理，独立随机变量的和趋于高斯分布。其次，高斯随机变量对于许多分析操作都很方便，因为许多涉及高斯分布的积分实际上都有简单的闭式解。在本小节的其余部分，我们将回顾多元高斯的一些有用性质。

给定随机向量$x \in R^{n}$服从多元高斯分布$x\sim\mathcal{N}(\mu,\Sigma)$。假设$x$中的变量被分成两个集合$x_{A}=\left[x_{1} \cdots x_{r}\right]^{T} \in R^{r}$和$x_{B}=\left[x_{r+1} \cdots x_{n}\right]^{T} \in R^{n-r}$（对于$\mu$和$\Sigma$也进行同样的拆分），则有：

$$
x=\left[ \begin{array}{c}{x_{A}} \\ {x_{B}}\end{array}\right] \qquad 
\mu=\left[ \begin{array}{c}{\mu_{A}} \\ {\mu_{B}}\end{array}\right] \qquad 
\Sigma=\left[ \begin{array}{cc}{\sum_{A A}} & {\sum_{A B}} \\ {\Sigma_{B A}} & {\Sigma_{B B}}\end{array}\right]
$$

因为$\Sigma=E\left[(x-\mu)(x-\mu)^{T}\right]=\Sigma^{T}$，所以上式中有$\Sigma_{A B}=\Sigma_{B A}^{T}$。下列性质适用：

1. **规范性。** 概率密度函数的归一化，即：

$$
\int_{x} p(x ; \mu, \Sigma) dx = 1
$$

这个特性乍一看似乎微不足道，但实际上对于计算各种积分非常有用，即使是那些看起来与概率分布完全无关的积分（参见附录A.1）！

2. **边缘性。** 边缘概率密度函数：

$$
\begin{aligned} p\left(x_{A}\right) &=\int_{x_{B}} p\left(x_{A} , x_{B} ; \mu, \Sigma\right) d x_{B} \\ 
p\left(x_{B}\right) &=\int_{x_{A}} p\left(x_{A}, x_{B} ; \mu,\Sigma\right) d x_{A} \end{aligned}
$$

是高斯分布：

$$
\begin{aligned} x_{A} & \sim \mathcal{N}\left(\mu_{A}, \Sigma_{A A}\right) \\ 
x_{B} & \sim \mathcal{N}\left(\mu_{B}, \Sigma_{B B}\right) \end{aligned}
$$

3. **条件性。** 条件概率密度函数：

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

#### 2. 贝叶斯线性回归

设$S=\left\{\left(x^{(i)}, y^{(i)}\right)\right\}_{i=1}^{m}$是一组来自未知分布的满足独立同分布的训练集。线性回归的标准概率解释的公式说明了这一点：

$$
y^{(i)}=\theta^{T} x^{(i)}+\varepsilon^{(i)}, \quad i=1, \dots, m
$$

其中$\varepsilon^{(i)}$是独立同分布的“噪声”变量并且服从分布$\mathcal{N}(0,\Sigma^2)$，由此可见$y^{(i)}-\theta^{T} x^{(i)} \sim \mathcal{N}\left(0, \sigma^{2}\right)$，或等价表示为：

$$
P\left(y^{(i)} | x^{(i)}, \theta\right)=\frac{1}{\sqrt{2 \pi} \sigma} \exp \left(-\frac{\left(y^{(i)}-\theta^{T} x^{(i)}\right)^{2}}{2 \sigma^{2}}\right)
$$

为了方便标记，我们定义了：

$$
X=\left[ \begin{array}{c}{-\left(x^{(1)}\right)^{T}-} \\ {-\left(x^{(2)}\right)^{T}-} \\ {\vdots} \\ {-\left(x^{(m)}\right)^{T}-}\end{array}\right] \in \mathbf{R}^{m \times n} \qquad 
\vec{y}=\left[ \begin{array}{c}{y^{(1)}} \\ {y^{(2)}} \\ {\vdots} \\ {y^{(m)}}\end{array}\right] \in \mathbf{R}^{m} \qquad 
\overrightarrow{\varepsilon}=\left[ \begin{array}{c}{\varepsilon^{(1)}} \\ {\varepsilon^{(2)}} \\ {\vdots} \\ {\varepsilon^{(m)}}\end{array}\right] \in \mathbf{R}^{m}
$$

在贝叶斯线性回归中，我们假设参数的**先验分布**也是给定的；例如，一个典型的选择是$\theta \sim \mathcal{N}\left(0, \tau^{2} I\right)$。使用贝叶斯规则可以得到**后验参数：**

$$
p(\theta | S)=\frac{p(\theta) p(S | \theta)}{\int_{\theta^{\prime}} p\left(\theta^{\prime}\right) p\left(S | \theta^{\prime}\right) d \theta^{\prime}}=\frac{p(\theta) \prod_{i=1}^{m} p\left(y^{(i)} | x^{(i)}, \theta\right)}{\int_{\theta^{\prime}} p\left(\theta^{\prime}\right) \prod_{i=1}^{m} p\left(y^{(i)} | x^{(i)}, \theta^{\prime}\right) d \theta^{\prime}}\qquad\qquad(2)
$$

假设测试点上的噪声模型与我们的训练点上的噪声模型相同，那么贝叶斯线性回归在一个新的测试点$x_*$上的“输出”不只是一个猜测$y_*$，而可能是输出的整个概率分布，称为**后验预测分布：**

$$
p\left(y_{*} | x_{*}, S\right)=\int_{\theta} p\left(y_{*} | x_{*}, \theta\right) p(\theta | S) d \theta \qquad\qquad(3)
$$

对于许多类型的模型，$(2)$和$(3)$中的积分是很难计算的，因此，我们经常使用近似的方法，例如MAP 估计（参见[正则化和模型选择](https://kivy-cn.github.io/Stanford-CS-229-CN/#/Markdown/cs229-notes5)的课程讲义）。

然而，在贝叶斯线性回归的情况下，积分实际上是可处理的！特别是对于贝叶斯线性回归，（在做了大量工作之后！）我们可以证明：

$$
\theta | S \sim \mathcal{N}\left(\frac{1}{\sigma^{2}} A^{-1} X^{T} \vec{y}, A^{-1}\right) \\ 
y_{*} | x_{*}, S \sim \mathcal{N}\left(\frac{1}{\sigma^{2}} x_{*}^{T} A^{-1} X^{T} \vec{y}, x_{*}^{T} A^{-1} x_{*}+\sigma^{2}\right)
$$

其中$A=\frac{1}{\sigma^{2}} X^{T} X+\frac{1}{\tau^{2}} I$。这些公式的推导有点复杂。$^6$但是从这些方程中，我们至少可以大致了解贝叶斯方法的含义：对于测试输入$x_*$，测试输出$y_*$的后验分布是高斯分布——这个分布反映了在我们预测$y_{*}=\theta^{T} x_{*}+\varepsilon_{*}$时，由$\epsilon_*$的随机性以及我们选择参数$\theta$的不确定而导致预测结果的不确定性。相反,古典概率线性回归模型直接从训练数据估计参数$\theta$，但没有提供估计估计这些参数的可靠性（参见图1）。

>6 有关完整的推导，可以参考[1]`注：参考资料[1]见文章最下方`。或者参考附录，其中给出了一些基于平方补全技巧的参数，请自己推导这个公式！

![](https://raw.githubusercontent.com/Kivy-CN/Stanford-CS-229-CN/master/img/cs229notegpf1.png)

图1：一维线性回归问题的贝叶斯线性回归$y^{(i)}=\theta x^{(i)}+\epsilon^{(i)}$，其中噪音独立同分布的服从$\epsilon^{(i)}\sim \mathcal{N}(0,1)$。绿色区域表示模型预测的$95%$置信区间。注意，绿色区域的（垂直）宽度在末端最大，但在中部最窄。这个区域反映了参数$\theta$估计的不确定性。与之相反，经典线性回归模型会显示一个等宽的置信区域，在输出中只反映噪声服从$\mathcal{N}(0,\sigma^2)$。

#### 3. 高斯过程

如第$1$节所述，多元高斯分布由于其良好的分析性质，对于实值变量的有限集合建模是有用的。高斯过程是多元高斯函数的推广，适用于无穷大小的实值变量集合。特别地，这个扩展将允许我们把高斯过程看作不仅仅是随机向量上的分布实际上是**随机函数**上的分布。

>7 令$\mathcal{H}$是一类$\mathcal{X}\rightarrow\mathcal{Y}$的函数映射。一个来自$\mathcal{H}$的随机函数$f(\cdot)$代表根据$\mathcal{H}$的概率分布随机从$\mathcal{H}$中选择一个函数。一个潜在的困惑是：你可能倾向于认为随机函数的输出在某种程度上是随机的；事实并非如此。一个随机函数$f(\cdot)$，一旦有概率的从$\mathcal{H}$中选择，则表示从输入$\mathcal{X}$到输出$\mathcal{Y}$的确定性映射。