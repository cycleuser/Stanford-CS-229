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

我们在本课程上半部分讨论的许多经典机器学习算法都符合以下模式：给定一组从未知分布中采样的独立同分布的示例训练样本集：

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
p(x;\mu,\Sigma)=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)\qquad\qquad(1)
$$

我们可以写作$x\sim\mathcal{N}(\mu,\Sigma)$。这里，回想一下线性代数的笔记中$S_{++}^n$指的是对称正定$n\times n$矩阵$^5$的空间。

>5 实际上，在某些情况下，我们需要处理的多元高斯分布的$\Sigma$是正半定而非正定（即，$\Sigma$不满秩）。在这种情况下，$\Sigma^{-1}$不存在，所以$(1)$式中给出的高斯概率密度函数的定义并不适用。例子可以参阅“[因子分析](https://kivy-cn.github.io/Stanford-CS-229-CN/#/Markdown/cs229-notes9)”课程讲义。

一般来说，高斯随机变量在机器学习和统计中非常有用，主要有两个原因。首先，它们在统计算法中建模“噪声”时非常常见。通常，噪声可以被认为是影响测量过程的大量独立的小随机扰动的累积；根据中心极限定理，独立随机变量的和趋于高斯分布。其次，高斯随机变量对于许多分析操作都很方便，因为许多涉及高斯分布的积分实际上都有简单的闭式解。在本小节的其余部分，我们将回顾多元高斯分布的一些有用性质。

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

对于许多类型的模型，$(2)$和$(3)$中的积分是很难计算的，因此，我们经常使用近似的方法，例如MAP估计（参见[正则化和模型选择](https://kivy-cn.github.io/Stanford-CS-229-CN/#/Markdown/cs229-notes5)的课程讲义）。

然而，在贝叶斯线性回归的情况下，积分实际上是可处理的！特别是对于贝叶斯线性回归，（在做了大量工作之后！）我们可以证明：

$$
\theta | S \sim \mathcal{N}\left(\frac{1}{\sigma^{2}} A^{-1} X^{T} \vec{y}, A^{-1}\right) \\ 
y_{*} | x_{*}, S \sim \mathcal{N}\left(\frac{1}{\sigma^{2}} x_{*}^{T} A^{-1} X^{T} \vec{y}, x_{*}^{T} A^{-1} x_{*}+\sigma^{2}\right)
$$

其中$A=\frac{1}{\sigma^{2}} X^{T} X+\frac{1}{\tau^{2}} I$。这些公式的推导有点复杂。$^6$但是从这些方程中，我们至少可以大致了解贝叶斯方法的含义：对于测试输入$x_*$，测试输出$y_*$的后验分布是高斯分布——这个分布反映了在我们预测$y_{*}=\theta^{T} x_{*}+\varepsilon_{*}$时，由$\epsilon_*$的随机性以及我们选择参数$\theta$的不确定而导致预测结果的不确定性。相反，古典概率线性回归模型直接从训练数据估计参数$\theta$，但没有提供估计这些参数的可靠性（参见图1）。

>6 有关完整的推导，可以参考[1]`注：参考资料[1]见文章最下方`。或者参考附录，其中给出了一些基于平方补全技巧的参数，请自己推导这个公式！

![](https://raw.githubusercontent.com/Kivy-CN/Stanford-CS-229-CN/master/img/cs229notegpf1.png)

图1：一维线性回归问题的贝叶斯线性回归$y^{(i)}=\theta x^{(i)}+\epsilon^{(i)}$，其中噪音独立同分布的服从$\epsilon^{(i)}\sim \mathcal{N}(0,1)$。绿色区域表示模型预测的$95\%$置信区间。注意，绿色区域的（垂直）宽度在末端最大，但在中部最窄。这个区域反映了参数$\theta$估计的不确定性。与之相反，经典线性回归模型会显示一个等宽的置信区域，在输出中只反映噪声服从$\mathcal{N}(0,\sigma^2)$。

#### 3. 高斯过程

如第$1$节所述，多元高斯分布由于其良好的分析性质，对于实值变量的有限集合建模是有用的。高斯过程是多元高斯函数的推广，适用于无穷大小的实值变量集合。特别地，这个扩展将允许我们把高斯过程看作不仅仅是随机向量上的分布，而实际上是**随机函数**上的分布。

>7 令$\mathcal{H}$是一类$\mathcal{X}\rightarrow\mathcal{Y}$的函数映射。一个来自$\mathcal{H}$的随机函数$f(\cdot)$代表根据$\mathcal{H}$的概率分布随机从$\mathcal{H}$中选择一个函数。一个潜在的困惑是：你可能倾向于认为随机函数的输出在某种程度上是随机的；事实并非如此。一个随机函数$f(\cdot)$，一旦有概率的从$\mathcal{H}$中选择，则表示从输入$\mathcal{X}$到输出$\mathcal{Y}$的确定性映射。

##### 3.1 有限域函数上的概率分布

要了解如何对函数上的概率分布进行参数化，请考虑下面的简单示例。设$\mathcal{X}=\left\{x_{1}, \dots, x_{m}\right\}$为任何有限元素集。现在，考虑集合$\mathcal{H}$，该集合代表所有可能的从$\mathcal{X}$到$R$的函数映射。例如，可以给出如下的函数$f_0(\cdot)\in\mathcal{H}$的例子：

$$
f_{0}\left(x_{1}\right)=5, \quad f_{0}\left(x_{2}\right)=2.3, \quad f_{0}\left(x_{2}\right)=-7, \quad \ldots, \quad f_{0}\left(x_{m-1}\right)=-\pi, \quad f_{0}\left(x_{m}\right)=8
$$

因为任意函数$f(\cdot) \in \mathcal{H}$的定义域仅有$m$个元素，所以我们可以简介的使用$m$维向量$\vec{f}=\left[f\left(x_{1}\right) \quad f\left(x_{2}\right) \quad \cdots \quad f\left(x_{m}\right)\right]^{T}$表达$f(\cdot)$。为了指定函数$f(\cdot) \in \mathcal{H}$上的概率分布，我们必须把一些“概率密度”与$\mathcal{H}$中的每个函数联系起来。一种自然的方法是利用函数$f(\cdot) \in \mathcal{H}$和他们的向量表示$\vec{f}$之间的一一对应关系。特别是，如果我们指定$\vec{f} \sim \mathcal{N}\left(\overrightarrow{\mu}, \sigma^{2} I\right)$，则意味着函数$f(\cdot)$上的概率分布，其中函数$f(\cdot)$的概率密度函数可以通过下面的式子给出：

$$
p(h)=\prod_{i=1}^{m} \frac{1}{\sqrt{2 \pi} \sigma} \exp \left(-\frac{1}{2 \sigma^{2}}\left(f\left(x_{i}\right)-\mu_{i}\right)^{2}\right)
$$

在上面的例子中，我们证明了有限域函数上的概率分布可以用函数输出$f\left(x_{1}\right), \ldots, f\left(x_{m}\right)$的有限数量的输入点$x_{1}, \dots, x_{m}$上的有限维多元高斯分布来表示。当定义域的大小可能是无穷大时，我们如何指定函数上的概率分布？为此，我们转向一种更奇特的概率分布类型，称为高斯过程。

##### 3.2 无限域函数上的概率分布

随机过程是随机变量的集合$\{f(x) : x \in \mathcal{X}\}$，其来自某个集合$\mathcal{X}$的元素索引，称为索引集。$^8$ **高斯过程**是一个随机过程，任何有限子集合的随机变量都有一个多元高斯分布。

>8 通常，当$\mathcal{X} = R$时，可以将标识符$x\in \mathcal{X}$解释为表示时间，因此变量$f(x)$表示随时间的随机量的时间演化。然而，在高斯过程回归模型中，将标识符集作为回归问题的输入空间。

特别是一组随机变量集合$\{f(x) : x \in \mathcal{X}\}$被称为来自于一个具有**平均函数**$m(\cdot)$和**协方差函数**$k(\cdot, \cdot)$的高斯过程，满足对于任意元素是$x_{1}, \ldots, x_{m} \in \mathcal{X}$有限集合，相关的有限随机变量集$f\left(x_{1}\right), \ldots, f\left(x_{m}\right)$具有如下分布：

$$
\left[ \begin{array}{c}{f\left(x_{1}\right)} \\ {\vdots} \\ {f\left(x_{m}\right)}\end{array}\right]\sim
\mathcal{N}\left(\left[ \begin{array}{c}{m\left(x_{1}\right)} \\ {\vdots} \\ {m\left(x_{m}\right)}\end{array}\right], \left[ \begin{array}{ccc}{k\left(x_{1}, x_{1}\right)} & {\cdots} & {k\left(x_{1}, x_{m}\right)} \\ {\vdots} & {\ddots} & {\vdots} \\ {k\left(x_{m}, x_{1}\right)} & {\cdots} & {k\left(x_{m}, x_{m}\right)}\end{array}\right]\right)
$$

我们用下面的符号来表示：

$$
f(\cdot) \sim \mathcal{G P}(m(\cdot), k(\cdot, \cdot))
$$

注意，均值函数和协方差函数的名称很恰当，因为上述性质意味着：

$$
\begin{aligned} m(x) &=E[x] \\ k\left(x, x^{\prime}\right) &=E\left[(x-m(x))\left(x^{\prime}-m\left(x^{\prime}\right)\right)\right.\end{aligned}
$$

对于任意$x,x'\in\mathcal{X}$。

直观地说，我们可以把从高斯过程中得到的函数$f(\cdot)$看作是由高维多元高斯函数得到的高维向量。这里，高斯函数的每个维数对应于标识符集合$\mathcal{X}$中的一个元素$x$，随机向量的对应分量表示$f(x)$的值。利用多元高斯函数的边缘性，我们可以得到任意有限子集合所对应的多元高斯函数的边缘概率密度函数。

什么样的函数$m(\cdot)$和$k(\cdot,\cdot)$才能产生有效的高斯过程呢？一般情况下，任何实值函数$m(\cdot)$都是可以接受的，但是对于$k(\cdot,\cdot)$，对于任何一组元素$x_{1}, \ldots, x_{m} \in \mathcal{X}$都必须是可以接受的，结果矩阵如下：

$$
K=\left[ \begin{array}{ccc}{k\left(x_{1}, x_{1}\right)} & {\cdots} & {k\left(x_{1}, x_{m}\right)} \\ {\vdots} & {\ddots} & {\vdots} \\ {k\left(x_{m}, x_{1}\right)} & {\cdots} & {k\left(x_{m}, x_{m}\right)}\end{array}\right]
$$

是一个有效的协方差矩阵，对应于某个多元高斯分布。概率论中的一个标准结果表明，如果$K$是正半定的，这是正确的。听起来是不是很熟悉？

基于任意输入点计算协方差矩阵的正半定条件，实际上与核的Mercer条件相同！函数$k(\cdot,\cdot)$是一个有效的核，前提是对于任意一组输入点$x_{1}, \ldots, x_{m} \in \mathcal{X}$，因此，任何有效的核函数都可以用作协方差函数，这就是基于核的概率分布。

##### 3.3 平方指数核

![](https://raw.githubusercontent.com/Kivy-CN/Stanford-CS-229-CN/master/img/cs229notegpf2.png)

图2：样本来自于一个零均值高斯过程，以$k_{S E}(\cdot, \cdot)$为先验协方差函数。使用(a) $\tau=0.5,$ (b) $\tau=2,$ and $(\mathrm{c}) \tau=10$。注意,随着带宽参数$\tau$的增加，然后点比以前更远会有较高的相关性，因此采样函数往往整体是流畅的。

为了直观地了解高斯过程是如何工作的，考虑一个简单的零均值高斯过程：

$$
f(\cdot) \sim \mathcal{G P}(0, k(\cdot, \cdot))
$$

定义一些函数$h:\mathcal{X}\rightarrow R$，其中$\mathcal{X}=R$。这里，我们选择核函数 作为**平方指数**$^9$核函数，定义如下：

>9 在支持向量机的背景下，我们称之为高斯核；为了避免与高斯过程混淆，我们将这个核称为平方指数核，尽管这两个核在形式上是相同的。

$$
k_{S E}\left(x, x^{\prime}\right)=\exp \left(-\frac{1}{2 \tau^{2}}\left\|x-x^{\prime}\right\|^{2}\right)
$$

对于一些$\tau> 0$。从这个高斯过程中采样的随机函数是什么样的？

在我们的例子中，由于我们使用的是一个零均值高斯过程，我们期望高斯过程中的函数值会趋向于分布在零附近。此外，对于任意一对元素$x, x^{\prime} \in \mathcal{X}$。

- $f(x)$和$f(x')$将趋向于有高协方差$x$和$x'$在输入空间“附近”（即：$\left\|x-x^{\prime}\right\|=\left|x-x^{\prime}\right| \approx 0,$ 因此 $\exp \left(-\frac{1}{2 \tau^{2}}\left\|x-x^{\prime}\right\|^{2}\right) \approx 1$）
- 当$x$和$x'$相距很远时，$f(x)$和$f(x')$的协方差很低（即：$\left\|x-x^{\prime}\right\| \gg 0,$ 因此$\exp \left(-\frac{1}{2 \tau^{2}}\left\|x-x^{\prime}\right\|^{2}\right) \approx 0$）

更简单地说，从一个零均值高斯过程中得到的函数具有平方指数核，它将趋向于局部光滑，具有很高的概率；即：附近的函数值高度相关，并且在输入空间中相关性作为距离的函数递减（参见图2）。

#### 4. 高斯过程回归

正如上一节所讨论的，高斯过程为函数上的概率分布提供了一种建模方法。在这里，我们讨论了如何在贝叶斯回归的框架下使用函数上的概率分布。

##### 4.1 高斯过程回归模型

![](https://raw.githubusercontent.com/Kivy-CN/Stanford-CS-229-CN/master/img/cs229notegpf3.png)

图3：高斯过程回归使用一个零均值高斯先验过程，以$k_{S E}(\cdot, \cdot)$为协方差函数（其中$\tau=0.1$），其中噪声等级为$\sigma=1$以及$(a)m=10,(b) m=20，(c)m=40$训练样本。蓝线表示后验预测分布的均值，绿色阴影区域表示基于模型方差估计的$95%$置信区间。随着训练实例数量的增加，置信区域的大小会缩小，以反映模型估计中不确定性的减少。还请注意，在图像$(a)$中，$95%$置信区间在训练点附近缩小，但在远离训练点的地方要大得多，正如人们所期望的那样。

设$S=\left\{\left(x^{(i)}, y^{(i)}\right)\right\}_{i=1}^{m}$是一组来自未知分布的满足独立同分布的训练集。在高斯过程回归模型中公式说明了这一点：

$$
y^{(i)}=f\left(x^{(i)}\right)+\varepsilon^{(i)}, \quad i=1, \ldots, m
$$

其中$\varepsilon^{(i)}$是独立同分布的“噪声”变量并且服从分布$\mathcal{N}(0,\Sigma^2)$。就像在贝叶斯线性回归中，我们也假设一个函数$f(\cdot)$的**先验分布。** 特别地，我们假设一个零均值高斯过程先验：

$$
f(\cdot) \sim \mathcal{G} \mathcal{P}(0, k(\cdot, \cdot))
$$

对于一些有效的协方差函数$k(\cdot, \cdot)$。

现在，设$T=\left\{\left(x_{*}^{(i)}, y_{*}^{(i)}\right)\right\}_{i=1}^{m_{*}}$是从一些未知分布$S$中取得的独立同分布的测试点集合。$^10$为了方便标记，我们定义：

>10 我们还假设$T$和$S$是相互独立的。

$$
X=
\left[ \begin{array}{c}{-\left(x^{(1)}\right)^{T}-} \\ {-\left(x^{(2)}\right)^{T}-} \\ {\vdots} \\ {-\left(x^{(m)}\right)^{T}-}\end{array}\right] \in \mathbf{R}^{m \times n} \quad 
\vec{f}=
\left[ \begin{array}{c}{f\left(x^{(1)}\right)} \\ {f\left(x^{(2)}\right)} \\ {\vdots} \\ {f\left(x^{(m)}\right)}\end{array}\right], \quad 
\overrightarrow{\varepsilon}=
\left[ \begin{array}{c}{\varepsilon^{(1)}} \\ {\varepsilon^{(2)}} \\ {\vdots} \\ {\varepsilon^{(m)}}\end{array}\right], \quad 
\vec{y}=
\left[ \begin{array}{c}{y^{(1)}} \\ {y^{(2)}} \\ {\vdots} \\ {y^{(m)}}\end{array}\right] \in \mathbf{R}^{m} \\
X_{*}=
\left[ \begin{array}{c}{-\left(x_{*}^{(1)}\right)^{T}-} \\ {-\left(x_{*}^{(2)}\right)^{T}-} \\ {\vdots} \\ {-\left(x_{*}^{\left(m_{*}\right)}\right)^{T}-}\end{array}\right] \in \mathbf{R}^{m_{*} \times n} \quad 
\overrightarrow{f_{*}}=
\left[ \begin{array}{c}{f\left(x_{*}^{(1)}\right)} \\ {f\left(x_{*}^{(2)}\right)} \\ {\vdots} \\ {f\left(x_{*}^{\left(m_{*}\right)}\right)}\end{array}\right], \quad
\overrightarrow{\varepsilon}_{*}=
\left[ \begin{array}{c}{\varepsilon_{*}^{(1)}} \\ {\varepsilon_{*}^{(2)}} \\ {\vdots} \\ {\varepsilon_{*}^{\left(m_{*}\right)}}\end{array}\right], \quad 
\vec{y}_{*}=
\left[ \begin{array}{c}{y_{*}^{(1)}} \\ {y_{*}^{(2)}} \\ {\vdots} \\ {y_{*}^{\left(m_{*}\right)}}\end{array}\right] \in \mathbf{R}^{m}
$$

给定训练数据$S$，先验$p(h)$，以及测试输入$X_*$，我们如何计算测试输出的后验预测分布？对于第$2$节中的贝叶斯线性回归，我们使用贝叶斯规则来计算后验参数，然后对于新的测试点$x_*$使用后验参数计算后验预测分布$p\left(y_{*} | x_{*}, S\right)$。然而，对于高斯过程回归，结果是存在一个更简单的解决方案！

##### 4.2 预测

回想一下，对于从具有协方差函数$k(\cdot,\cdot)$的零均值高斯先验过程中得到的任何函数$f(\cdot)$，其任意一组输入点上的边缘分布必须是一个联合的多元高斯分布。特别是，这必须适用于训练和测试点，所以我们有下式：

$$
\left[ \begin{array}{c}{\vec{f}} \\ {\vec{f}_*}\end{array}\right] | X, X_{*} \sim \mathcal{N}\left(\overrightarrow{0}, \left[ \begin{array}{cc}{K(X, X)} & {K\left(X, X_{*}\right)} \\ {K\left(X_{*}, X\right)} & {K\left(X_{*}, X_{*}\right)}\end{array}\right]\right)
$$

其中：

$$
\vec{f} \in \mathbf{R}^{m} \text { such that } \vec{f}=\left[f\left(x^{(1)}\right) \cdots f\left(x^{(m)}\right)\right]^{T}\\
\vec{f}_{*} \in \mathbf{R}^{m} \cdot \text { such that } \vec{f}_{*}=\left[f\left(x_{*}^{(1)}\right) \cdots f\left(x_{*}^{(m)}\right)\right]^{T} \\
K(X, X) \in \mathbf{R}^{m \times m} \text { such that }(K(X, X))_{i j}=k\left(x^{(i)}, x^{(j)}\right) \\
K\left(X, X_{*}\right) \in \mathbf{R}^{m \times m_*} \text { such that }\left(K\left(X, X_{*}\right)\right)_{i j}=k\left(x^{(i)}, x_{*}^{(j)}\right) \\
K\left(X_{*}, X\right) \in \mathbf{R}^{m_* \times m} \text { such that }\left(K\left(X_{*}, X\right)\right)_{i j}=k\left(x_{*}^{(i)}, x^{(j)}\right) \\
K\left(X_{*}, X_{*}\right) \in \mathbf{R}^{m_{*} \times m_{*}} \text { such that }\left(K\left(X_{*}, X_{*}\right)\right)_{i j}=k\left(x_{*}^{(i)}, x_{*}^{(j)}\right)
$$

根据我们独立同分布噪声假设，可以得到：

$$
\left[ \begin{array}{c}{\overrightarrow{\varepsilon}} \\ {\overrightarrow{\varepsilon}_{*}}\end{array}\right]\sim\mathcal{N}\left(0,\left[ \begin{array}{cc}{\sigma^{2} I} & {\overrightarrow{0}} \\ {\overrightarrow{0}^{T}} & {\sigma^{2} I}\end{array}\right]\right)
$$

独立高斯随机变量的和也是高斯的，所以有：

$$
\left[ \begin{array}{c}{\vec{y}} \\ {\vec{y}_{*}}\end{array}\right] | X, X_{*}=
\left[ \begin{array}{c}{\vec{f}} \\ {\vec{f}}\end{array}\right]+\left[ \begin{array}{c}{\overrightarrow{\varepsilon}} \\ {\overrightarrow{\varepsilon}_{*}}\end{array}\right] \sim 
\mathcal{N}\left(\overrightarrow{0}, \left[ \begin{array}{cc}{K(X, X)+\sigma^{2} I} & {K\left(X, X_{*}\right)} \\ {K\left(X_{*}, X\right)} & {K\left(X_{*}, X_{*}\right)+\sigma^{2} I}\end{array}\right]\right)
$$

现在，用高斯函数的条件设定规则，它遵循下面的式子：

$$
\overrightarrow{y_{*}} | \vec{y}, X, X_{*} \sim \mathcal{N}\left(\mu^{*}, \Sigma^{*}\right)
$$

其中：

$$
\begin{aligned} \mu^{*} &=K\left(X_{*}, X\right)\left(K(X, X)+\sigma^{2} I\right)^{-1} \vec{y} \\
\Sigma^{*} &=K\left(X_{*}, X_{*}\right)+\sigma^{2} I-K\left(X_{*}, X\right)\left(K(X, X)+\sigma^{2} I\right)^{-1} K\left(X, X_{*}\right) \end{aligned}
$$

就是这样！值得注意的是，在高斯过程回归模型中进行预测非常简单，尽管高斯过程本身相当复杂！$^{11}$

>11 有趣的是，贝叶斯线性回归，当以正确的方式进行核化时，结果与高斯过程回归完全等价！但贝叶斯线性回归的后验预测分布的推导要复杂得多，对算法进行核化的工作量更大。高斯过程透视图当然要简单得多。

#### 5. 总结

在结束对高斯过程的讨论时，我们指出了高斯过程在回归问题中是一个有吸引力的模型的一些原因，在某些情况下，高斯过程可能优于其他模型（如线性和局部加权线性回归）：

1. 作为贝叶斯方法，高斯过程模型不仅可以量化问题的内在噪声，还可以量化参数估计过程中的误差，从而使预测的不确定性得到量化。此外，贝叶斯方法中的许多模型选择和超参数选择方法都可以立即应用于高斯过程（尽管我们没有在这里讨论这些高级主题）。
2. 与局部加权线性回归一样，高斯过程回归是非参数的，因此可以对输入点的任意函数进行建模。
3. 高斯过程回归模型为将核引入回归建模框架提供了一种自然的方法。通过对核的仔细选择，高斯过程回归模型有时可以利用数据中的结构（尽管我们也没有在这里研究这个问题）。
4. 高斯过程回归模型，尽管在概念上可能有些难以理解，但仍然导致了简单而直接的线性代数实现。

##### 参考资料

>[1] Carl E. Rasmussen and Christopher K. I. Williams. Gaussian Processes for Machine Learning. MIT Press, 2006. Online: [http://www.gaussianprocess.org/gpml/](http://www.gaussianprocess.org/gpml/)

##### 附录 A.1

在这个例子中，我们展示了如何使用多元高斯的归一化特性来计算相当吓人的多元积分，而不需要执行任何真正的微积分！假设你想计算下面的多元积分：

$$
I(A, b, c)=\int_{x} \exp \left(-\frac{1}{2} x^{T} A x-x^{T} b-c\right) d x
$$

尽管可以直接执行多维积分（祝您好运！），但更简单的推理是基于一种称为“配方法”的数学技巧。特别的：

$$
\begin{aligned} I(A, b, c) 
&=\exp (-c) \cdot \int_{x} \exp \left(-\frac{1}{2} x^{T} A x-x^{T} A A^{-1} b\right)d x \\ 
&=\exp (-c) \cdot \int_{x} \exp \left(-\frac{1}{2}\left(x-A^{-1} b\right)^{T} A\left(x-A^{-1} b\right)-b^{T} A^{-1} b\right) d x \\ 
&=\exp \left(-c-b^{T} A^{-1} b\right) \cdot \int_{x} \exp \left(-\frac{1}{2}\left(x-A^{-1} b\right)^{T} A\left(x-A^{-1} b\right)\right) d x \end{aligned}
$$

定义$\mu=A^{-1} b$ 和 $\Sigma=A^{-1}$，可以得到$I(A,b,c)$等于：

$$
\frac{(2 \pi)^{m / 2}|\Sigma|^{1 / 2}}{\exp \left(c+b^{T} A^{-1} b\right)} \cdot\left[\frac{1}{(2 \pi)^{m / 2}|\Sigma|^{1 / 2}} \int_{x} \exp \left(-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right) d x\right]
$$

然而，括号中的项在形式上与多元高斯函数的积分是相同的！因为我们知道高斯密度可以归一化，所以括号里的项等于$1$。因此：

$$
I(A, b, c)=\frac{(2 \pi)^{m / 2}\left|A^{-1}\right|^{1 / 2}}{\exp \left(c+b^{T} A^{-1} b\right)}
$$

##### 附录 A.2

推导出给定$x_B$下$x_A$的分布形式；另一个结果可以立即根据对称性可以得到。注意到：

$$
\begin{aligned}
p\left(x_{A} | x_{B}\right)&=\frac{1}{\int_{x_{A}} p\left(x_{A}, x_{B} ; \mu, \Sigma\right) d x_{A}} \cdot\left[\frac{1}{(2 \pi)^{m / 2}|\Sigma|^{1 / 2}} \exp \left(-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right)\right] \\
&=\frac{1}{Z_{1}} \exp \left\{-\frac{1}{2}\left(\left[ \begin{array}{c}{x_{A}} \\ {x_{B}}\end{array}\right]-\left[ \begin{array}{c}{\mu_{A}} \\ {\mu_{B}}\end{array}\right]\right)^{T} \left[ \begin{array}{cc}{V_{A A}} & {V_{A B}} \\ {V_{B A}} & {V_{B B}}\end{array}\right]\left(\left[ \begin{array}{c}{x_{A}} \\ {x_{B}}\end{array}\right]-\left[ \begin{array}{c}{\mu_{A}} \\ {\mu_{B}}\end{array}\right]\right)\right\}
\end{aligned}
$$

其中$Z_1$是不依赖于$x_A$的比例常数，且：

$$
\Sigma^{-1}=V=\left[ \begin{array}{ll}{V_{A A}} & {V_{A B}} \\ {V_{B A}} & {V_{B B}}\end{array}\right]
$$

要简化这个表达式，请观察下面的式子：

$$
\begin{aligned}
&\left(\left[ \begin{array}{c}{x_{A}} \\ {x_{B}}\end{array}\right]-\left[ \begin{array}{c}{\mu_{A}} \\ {\mu_{B}}\end{array}\right]\right)^{T} \left[ \begin{array}{cc}{V_{A A}} & {V_{A B}} \\ {V_{B A}} & {V_{B B}}\end{array}\right]\left(\left[ \begin{array}{c}{x_{A}} \\ {x_{B}}\end{array}\right]-\left[ \begin{array}{c}{\mu_{A}} \\ {\mu_{B}}\end{array}\right]\right) \\
&\qquad =\left(x_{A}-\mu_{A}\right)^{T} V_{A A}\left(x_{A}-\mu_{A}\right)+\left(x_{A}-\mu_{A}\right)^{T} V_{A B}\left(x_{B}-\mu_{B}\right) \\
&\qquad\qquad +\left(x_{B}-\mu_{B}\right)^{T} V_{B A}\left(x_{A}-\mu_{A}\right)+\left(x_{B}-\mu_{B}\right)^{T} V_{B B}\left(x_{B}-\mu_{B}\right)
\end{aligned}
$$

只保留依赖于$x_A$的项（利用$V_{A B}=V_{B A}^{T}$），我们有：

$$
p\left(x_{A} | x_{B}\right)=\frac{1}{Z_{2}} \exp \left(-\frac{1}{2}\left[x_{A}^{T} V_{A A} x_{A}-2 x_{A}^{T} V_{A A} \mu_{A}+2 x_{A}^{T} V_{A B}\left(x_{B}-\mu_{B}\right)\right]\right)
$$

其中$Z_2$是一个同样不依赖于$x_A$新的比例常数。最后，使用“配方”参数（参见附录A.1），我们得到：

$$
p\left(x_{A} | x_{B}\right)=\frac{1}{Z_{3}} \exp \left(-\frac{1}{2}\left(x_{A}-\mu^{\prime}\right)^{T} V_{A A}\left(x_{A}-\mu^{\prime}\right)\right)
$$

其中$Z_3$是一个新的不依赖于$x_A$的比例常数，并且$\mu'=\mu_{A}-V_{A A}^{-1} V_{A B}\left(x_{B}-\mu_{B}\right)$。最后这个表述表明以$x_B$为条件下$x_A$的分布，同样是多元高斯函数的形式。事实上，从归一化性质可以直接得出：

$$
x_{A} | x_{B} \sim \mathcal{N}\left(\mu_{A}-V_{A A}^{-1} V_{A B}\left(x_{B}-\mu_{B}\right), V_{A A}^{-1}\right)
$$

为了完成证明，我们只需要注意：

$$
\left[ \begin{array}{cc}{V_{A A}} & {V_{A B}} \\ {V_{B A}} & {V_{B B}}\end{array}\right]=
\left[ \begin{array}{c}{\left(\Sigma_{A A}-\Sigma_{A B} \Sigma_{B B}^{-1} \Sigma_{B A}\right)^{-1}}&-\left(\Sigma_{A A}-\Sigma_{A B} \Sigma_{B B}^{-1} \Sigma_{B A}\right)^{-1} \Sigma_{A B} \Sigma_{B B}^{-1} \\ {-\Sigma_{B B}^{-1} \Sigma_{B A}\left(\Sigma_{A A}-\Sigma_{A B} \Sigma_{B B}^{-1} \Sigma_{B A}\right)^{-1}}&\left(\Sigma_{B B}-\Sigma_{B A} \Sigma_{A A}^{-1} \Sigma_{A B}\right)^{-1}\end{array} \right]
$$

由分块矩阵的逆的标准公式推出。将相关的块替换到前面的表达式中就得到了想要的结果。

##### 附录 A.3

在这一节中，我们提出了多元高斯分布条件分布的另一种（更简单的）推导方法。注意，正如附录A.2所示，我们可以这样写出$p\left(x_{A} | x_{B}\right)$的形式：

$$
\begin{aligned} 
p\left(x_{A} | x_{B}\right) 
&=\frac{1}{\int_{x_{A}} p\left(x_{A}, x_{B} ; \mu, \Sigma\right) d x_{A}} \cdot\left[\frac{1}{(2 \pi)^{m / 2}|\Sigma|^{1 / 2}} \exp \left(-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right)\right] &(4)\\ 
&=\frac{1}{Z_{1}} \exp \left\{-\frac{1}{2}\left(\left[ \begin{array}{c}{x_{A}-\mu_{A}} \\ {x_{B}-\mu_{B}}\end{array}\right]\right)^{T} \left[ \begin{array}{cc}{V_{A A}} & {V_{A B}} \\ {V_{B A}} & {V_{B B}}\end{array}\right] \left[ \begin{array}{c}{x_{A}-\mu_{A}} \\ {x_{B}-\mu_{B}}\end{array}\right]\right\} &(5)
\end{aligned}
$$

其中$Z_1$是不依赖于$x_A$的比例常数。

这个推导使用了一个附加的假设，即条件分布是一个多元高斯分布；换句话说，我们假设$p\left(x_{A} | x_{B}\right) \sim \mathcal{N}\left(\mu^{*}, \Sigma^{*}\right)$有一些参数$\mu^{*}, \Sigma^{*}$（或者，你可以把这个推导看作是寻找“配方法”另一种方法）。

这个推导的关键直觉是当$x_{A}=\mu^{*} \triangleq x_{A}^{*}$时，$p\left(x_{A} | x_{B}\right)$将会最大化。我们计算$\log p\left(x_{A} | x_{B}\right)$关于$x_A$的梯度，并设其为零。利用等式$(5)$，我们可以得到：

$$
\begin{aligned}
&\nabla_{x_{A}} \log p(x_A | x_B)|_{x_A=x_A^{*}}  &\qquad\qquad\qquad(6)\\ 
&{=-V_{A A}\left(x_{A}^{*}-\mu_{A}\right)-V_{A B}\left(x_{B}-\mu_{B}\right)} &(7)\\ 
&{=0}&(8)
\end{aligned}
$$

这意味着：

$$
\mu^{*}=x_{A}^{*}=\mu_{A}-V_{A A}^{-1} V_{A B}\left(x_{B}-\mu_{B}\right)\qquad\qquad\qquad\qquad (9)
$$

类似地，我们利用高斯分布$p(\cdot)$的逆协方差矩阵是$\log p(\cdot)$的负海森矩阵。换句话说，高斯分布$p\left(x_{A} | x_{B}\right)$的逆协方差矩阵是$\log p\left(x_{A} | x_{B}\right)$的负海森矩阵。利用式$(5)$，我们有：

$$
\begin{aligned} 
\Sigma^{*-1} &=-\nabla_{x_{A}} \nabla_{x_{A}}^{T} \log p\left(x_{A} | x_{B}\right)&\qquad\qquad\qquad(10) \\ 
&=V_{A A} &(11)
\end{aligned}
$$

因此，我们得到：

$$
\Sigma^{*}=V_{A A}^{-1} \qquad\qquad\qquad(11)
$$
