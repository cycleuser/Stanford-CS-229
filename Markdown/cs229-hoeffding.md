# CS229 课程讲义中文翻译
CS229 Supplementary notes

|原作者|翻译|
|---|---|
|John Duchi|[XiaoDong_Wang](https://github.com/Dongzhixiao) |


|相关链接|
|---|
|[Github 地址](https://github。com/Kivy-CN/Stanford-CS-229-CN)|
|[知乎专栏](https://zhuanlan。zhihu。com/MachineLearn)|
|[斯坦福大学 CS229 课程网站](http://cs229。stanford。edu/)|
|[网易公开课中文字幕视频](http://open。163。com/movie/2008/1/M/C/M6SGF6VB4_M6SGHFBMC。html)|


### Hoeffding不等式

#### 1. 基本概率边界

概率论、统计学和机器学习的一个基本问题是：给定一个期望为$E[Z]$的随机变量$Z$，$Z$接近其期望的可能性有多大？更准确地说，它有多接近的概率是多少？考虑到这一点，本节内容提供了一些计算型如下面公式边界的方法：

$$
\mathbb{P}(Z \geq \mathbb{E}[Z]+t) \text { and } \mathbb{P}(Z \leq \mathbb{E}[Z]-t)\qquad\qquad(1)
$$

其中$t\ge 0$。

我们的第一个边界可能是所有概率不等式中最基本的，它被称为马尔可夫不等式。考虑到它的基本性质，其证明基本上只有一行就不足为奇了。

**命题1**（马尔可夫不等式）令$z\ge 0$是一个非负随机变量。则对于所有$t\ge 0$有：

$$
\mathbb{P}(Z \geq t) \leq \frac{\mathbb{E}[Z]}{t}
$$

**证明** 我们注意到$\mathbb{P}(Z \geq t)=\mathbb{E}[1\{Z \geq | t\}]$，以及如果$z\ge t$，则一定可得$Z / t \geq 1 \geq 1\{Z \geq t\}$，然而如果$z < t$，则我们仍然有$Z / t \geq 0=1\{Z \geq t\}$。因此：

$$
\mathbb{P}(Z \geq t)=\mathbb{E}[1\{Z \geq t\}] \leq \mathbb{E}\left[\frac{Z}{t}\right]=\frac{\mathbb{E}[Z]}{t}
$$

跟希望的一样。

本质上，$(1)$式概率上的所有其他边界都是马尔可夫不等式的变化。第一个变量用二阶矩表示随机变量的方差，而不是简单的均值，称为Chebyshev不等式。

**命题2** （切比雪夫不等式）。设$Z$为$Var(Z) < 1$的任意随机变量。则：

$$
\mathbb{P}(Z \geq \mathbb{E}[Z]+t \text { or } Z \leq \mathbb{E}[Z]-t) \leq \frac{\operatorname{Var}(Z)}{t^{2}}
$$

对于$t\ge 0$。

**证明** 这个结果是马尔可夫不等式的直接结果。我们注意到如果$Z \geq \mathbb{E}[Z]+t$，则我们一定能得到$(Z-\mathbb{E}[Z])^{2} \geq t^{2}$，并且类似的如果$Z \leq \mathbb{E}[Z]-t$，我们有$(Z-\mathbb{E}[Z])^{2} \geq t^{2}$。因此：

$$
\begin{aligned}
\mathbb{P}(Z \geq \mathbb{E}[Z]+t \text { or } Z \leq \mathbb{E}[Z]-t) 
&=\mathbb{P}\left((Z-\mathbb{E}[Z])^{2} \geq t^{2}\right) \\ 
& \stackrel{(i)}{ \leq} \frac{\mathbb{E}\left[(Z-\mathbb{E}[Z])^{2}\right]}{t^{2}}=\frac{\operatorname{Var}(Z)}{t^{2}} 
\end{aligned}
$$

其中步骤$(i)$是马尔可夫不等式。

切比雪夫不等式的一个很好的结果是有限方差随机变量的均值收敛于它们的均值。让我们举个例子。假设$Z_i$是独立同分布的，并满足$\mathbb{E}\left[Z_{i}\right]=0$。则$\mathbb{E}\left[Z_{i}\right]=0$，如果我们定义$\overline{Z}=\frac{1}{n} \sum_{i=1}^{n} Z_{i}$，则：

$$
\operatorname{Var}(\overline{Z})=\mathbb{E}\left[\left(\frac{1}{n} \sum_{i=1}^{n} Z_{i}\right)^{2}\right]=\frac{1}{n^{2}} \sum_{i, j \leq n} \mathbb{E}\left[Z_{i} Z_{j}\right]=\frac{1}{n^{2}} \sum_{i=1}^{n} \mathbb{E}\left[Z_{i}^{2}\right]=\frac{\operatorname{Var}\left(Z_{1}\right)}{n}
$$

特别的，对于任意$t\ge 0$，我们有：

$$
\mathbb{P}\left(\left|\frac{1}{n} \sum_{i=1}^{n} Z_{i}\right| \geq t\right) \leq \frac{\operatorname{Var}\left(Z_{1}\right)}{n t^{2}}
$$

因此对于任意$t > 0$有$\mathbb{P}(|\overline{Z}| \geq t) \rightarrow 0$。

#### 2. 矩母函数(Moment generating functions)

通常，我们希望对随机变量$Z$超出其期望的概率有更精确的（甚至是指数）边界。考虑到这一点，我们需要一个比有限方差更强的条件，对于有限方差，矩母函数是自然的候选条件。（方便的是，我们将看到它们也能很好地处理和。）回忆一下，对于随机变量$Z$, $Z$的矩母函数是下面这个函数：

$$
M_{Z}(\lambda) :=\mathbb{E}[\exp (\lambda Z)]
$$

其中对于一些$\lambda$来说是无限的。