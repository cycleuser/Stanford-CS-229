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
M_{Z}(\lambda) :=\mathbb{E}[\exp (\lambda Z)]\qquad\qquad(2)
$$

其中对于一些$\lambda$来说是无限的。

##### 2.1 切尔诺夫边界

切尔诺夫边界利用矩母函数的基本方法给出指数偏差界限。

**命题3** （切尔诺夫边界）。设$Z$为任意随机变量。然后对任何$t\ge 0$有：

$$
\mathbb{P}(Z \geq \mathbb{E}[Z]+t) \leq \min _{\lambda \geq 0} \mathbb{E}\left[e^{\lambda(Z-\mathbb{E}[Z])}\right] e^{-\lambda t}=\min _{\lambda \geq 0} M_{Z-\mathbb{E}[Z]}(\lambda) e^{-\lambda t}
$$

以及：

$$
\mathbb{P}(Z \leq \mathbb{E}[Z]-t) \leq \min _{\lambda \geq 0} \mathbb{E}\left[e^{\lambda(\mathbb{E}[Z]-Z)}\right] e^{-\lambda t}=\min _{\lambda \geq 0} M_{\mathbb{E}[Z]-Z}(\lambda) e^{-\lambda t}
$$

**证明** 我们只证明了第一个不等式，因为第二个不等式是完全等价的。我们使用马尔可夫不等式。对于任意$\lambda > 0$，当且仅当$e^{\lambda Z} \geq e^{\lambda \mathbb{E}[Z]+\lambda t}$或$e^{\lambda(Z-\mathbb{E}[Z])} \geq e^{\lambda t}$时我们有$Z \geq \mathbb{E}[Z]+t$。因此可得：

$$
\mathbb{P}(Z-\mathbb{E}[Z] \geq t)=\mathbb{P}\left(e^{\lambda(Z-\mathbb{E}[Z])} \geq e^{\lambda t}\right) \stackrel{(i)}{ \leq} \mathbb{E}\left[e^{\lambda(Z-\mathbb{E}[Z])}\right] e^{-\lambda t}
$$

其中不等式$(i)$来自马尔科夫不等式。既然我们选择的$\lambda > 0$无关紧要，因此我们可以通过最小化边界的右边来得到最好的一个。（要注意的是，这个界限在$\lambda = 0$处是成立的。）

重要的结果是切诺夫边界很好地处理了求和，这是矩母函数的结果。假设$Z_i$是独立的。则我们可得：

$$
M_{Z_{1}+\cdots+Z_{n}}(\lambda)=\prod_{i=1}^{n} M_{Z_{i}}(\lambda)
$$

这是因为：

$$
\mathbb{E}\left[\exp \left(\lambda \sum_{i=1}^{n} Z_{i}\right)\right]=\mathbb{E}\left[\prod_{i=1}^{n} \exp \left(\lambda Z_{i}\right)\right]=\prod_{i=1}^{n} \mathbb{E}\left[\exp \left(\lambda Z_{i}\right)\right]
$$

由于$Z_i$是独立的。这意味着当我们计算一个独立同分布变量和的切尔诺夫边界界时，我们只需要计算其中一个变量的矩母函数。事实上，假设$Z_i$是独立同分布的，并且（为了简单起见）均值为零。则可得：

$$
\begin{aligned} 
\mathbb{P}\left(\sum_{i=1}^{n} Z_{i} \geq t\right) 
& \leq \frac{\prod_{i=1}^{n} \mathbb{E}\left[\exp \left(\lambda Z_{i}\right)\right]}{e^{\lambda t}} \\ 
&=\left(\mathbb{E}\left[e^{\lambda Z_{1}}\right]\right)^{n} e^{-\lambda t} 
\end{aligned}
$$

根据切尔诺夫边界。

##### 2.2 矩母函数例子

现在我们给出几个矩母函数的例子，这些例子能让我们得到一些很好的偏差不等式。后面我们给出的所有例子都可以用如下的非常方便的边界形式：

$$
M_{Z}(\lambda)=\mathbb{E}\left[e^{\lambda Z}\right] \leq \exp \left(\frac{C^{2} \lambda^{2}}{2}\right) \text { for all } \lambda \in \mathbb{R}
$$

对于一些$C\in \mathbb{R}$（这取决于$Z$的分布）；这个形式非常适合应用切尔诺夫边界。

我们从经典正态分布开始，其中$Z \sim \mathcal{N}\left(0, \sigma^{2}\right)$。则我们可得：

$$
\mathbb{E}[\exp (\lambda Z)]=\exp \left(\frac{\lambda^{2} \sigma^{2}}{2}\right)
$$

我们省略了计算过程。（如果你好奇的话，应该自己把它算出来！）

第二个例子使用的是Rademacher随机变量，或者随机符号变量。令$S=1$有概率$\frac 12$，$S=-1$有概率$\frac 12$，则我们可以声明：

$$
\mathbb{E}\left[e^{\lambda S}\right] \leq \exp \left(\frac{\lambda^{2}}{2}\right) \quad \text { for all } \lambda \in \mathbb{R}\qquad\qquad(3)
$$

要理解不等式$(3)$，我们来用指数函数的泰勒展开式，即$e^{x}=\sum_{k=0}^{\infty} \frac{x^{k}}{k !}$。注意当$k$是奇数的时候$\mathbb{E}\left[S^{k}\right]=0$。当$k$是偶数的时候$\mathbb{E}\left[S^{k}\right]=1$。则我们可得：

$$
\begin{aligned} 
\mathbb{E}\left[e^{\lambda S}\right] &=\sum_{k=0}^{\infty} \frac{\lambda^{k} \mathbb{E}\left[S^{k}\right]}{k !} \\ 
&=\sum_{k=0,2,4, \ldots} \frac{\lambda^{k}}{k !}=\sum_{k=0}^{\infty} \frac{\lambda^{2 k}}{(2 k) !} 
\end{aligned}
$$

最后，对于所有的$k=0,1,2, \ldots$我们使用$(2 k) ! \geq 2^{k} \cdot k !$。因此：

$$
\mathbb{E}\left[e^{\lambda S}\right] \leq \sum_{k=0}^{\infty} \frac{\left(\lambda^{2}\right)^{k}}{2^{k} \cdot k !}=\sum_{k=0}^{\infty}\left(\frac{\lambda^{2}}{2}\right)^{k} \frac{1}{k !}=\exp \left(\frac{\lambda^{2}}{2}\right)
$$

让我们把不等式$(3)$代入到一个切尔诺夫函数中，看看独立同分布随机符号的和有多大。

我们知道如果$Z=\sum_{i=1}^{n} S_{i}$，其中$S_{i} \in\{ \pm 1\}$是随机符号，则$\mathbb{E}[Z]=0$。根据切尔诺夫边界，很明显可得：

$$
\mathbb{P}(Z \geq t) \leq \mathbb{E}\left[e^{\lambda Z}\right] e^{-\lambda t}=\mathbb{E}\left[e^{\lambda S_{1}}\right]^{n} e^{-\lambda t} \leq \exp \left(\frac{n \lambda^{2}}{2}\right) e^{-\lambda t}
$$

应用切诺夫边界定理，我们可以在$\lambda \geq 0$时将其最小化，等价于下式：

$$
\min _{\lambda \geq 0}\left\{\frac{n \lambda^{2}}{2}-\lambda t\right\}
$$

幸运的是，这是一个很容易最小化的函数，对该函数求导并将其设为零，我们可得$n \lambda-t=0$或$\lambda=t / n$，这样给出了：

$$
\mathbb{P}(Z \geq t) \leq \exp \left(-\frac{t^{2}}{2 n}\right)
$$

特别的，取$t=\sqrt{2 n \log \frac{1}{\delta}}$，我们可得：

$$
\mathbb{P}\left(\sum_{i=1}^{n} S_{i} \geq \sqrt{2 n \log \frac{1}{\delta}}\right) \leq \delta
$$

因此有很高的概率得出$Z=\sum_{i=1}^{n} S_{i}=O(\sqrt{n})$——$n$个独立随机符号的和基本上不会大于$O(\sqrt{n})$。

#### 3. Hoeffding引理和Hoeffding不等式

Hoeffding不等式是一种强大的技术，它可能是学习理论中最重要的不等式，用于确定有界随机变量和过大或过小的概率的边界。我们先给出该不等式，然后我们将证明它是基于我们之前的矩母函数计算的一个弱化版本。

**定理 4（Hoeffding不等式）** 令$Z_{1}, \ldots, Z_{n}$是独立边界随机变量，并且对于所有$i$有$Z_{i} \in[a, b]$，其中$-\infty< a \leq b < \infty$。则：

$$
\mathbb{P}\left(\frac{1}{n} \sum_{i=1}^{n}\left(Z_{i}-\mathbb{E}\left[Z_{i}\right]\right) \geq t\right) \leq \exp \left(-\frac{2 n t^{2}}{(b-a)^{2}}\right)
$$

以及：

$$
\mathbb{P}\left(\frac{1}{n} \sum_{i=1}^{n}\left(Z_{i}-\mathbb{E}\left[Z_{i}\right]\right) \leq-t\right) \leq \exp \left(-\frac{2 n t^{2}}{(b-a)^{2}}\right)
$$

对于所有$t\ge 0$都成立。

我们用$(1)$切诺夫边界和$(2)$一个经典引理Hoeffding引理的组合来证明定理$4$，我们现在陈述这个引理。

**引理 5（Hoeffding引理）** 令$Z$是边界随机变量，有$z\in [a,b]$。则：

$$
\mathbb{E}[\exp (\lambda(Z-\mathbb{E}[Z]))] \leq \exp \left(\frac{\lambda^{2}(b-a)^{2}}{8}\right) \quad \text { for all } \lambda \in \mathbb{R}
$$

**证明** 我们证明了这个引理的一个稍微弱一点的版本，它的因子是$2$而不是$8$，我们使用随机符号矩来生成边界和一个不相等性，称为Jensen不等式（我们将在后面的[EM算法](https://kivy-cn.github.io/Stanford-CS-229-CN/#/Markdown/cs229-notes8)推导中看到这个非常重要的不等式）。Jensen不等式表述如下：如果$f : \mathbb{R} \rightarrow \mathbb{R}$是一个凸函数，即$f$是碗型函数，则：

$$
f(\mathbb{E}[Z]) \leq \mathbb{E}[f(Z)]
$$



我们在概率论中使用了一种称为对称化的聪明技术来给出我们的结果（可能并不希望知道这一点，但它是概率论、机器学习和统计中非常常见的技术，所以学习它很好）。首先，让$Z'$是具有相同分布的随机变量$Z$的一个独立副本，使得$Z^{\prime} \in[a, b]$和$\mathbb{E}\left[Z^{\prime}\right]=\mathbb{E}[Z]$，但$Z$和$Z'$是独立的。则：

$$
\mathbb{E}_{Z}\left[\exp \left(\lambda\left(Z-\mathbb{E}_{Z}[Z]\right)\right)\right]=\mathbb{E}_{Z}\left[\exp \left(\lambda\left(Z-\mathbb{E}_{Z^{\prime}}\left[Z^{\prime}\right]\right)\right)\right] \stackrel{(i)}{ \leq} \mathbb{E}_{Z}\left[\mathbb{E}_{Z^{\prime}} \exp \left(\lambda\left(Z-Z^{\prime}\right)\right)\right]
$$

其中$\mathbb{E}_{Z}$和$\mathbb{E}_{Z'}$代表$Z$和$Z'$的期望。之后，在步骤$(i)$中将Jensen不等式应用到函数$f(x)=e^{-x}$上。现在我们有：

$$
\mathbb{E}[\exp (\lambda(Z-\mathbb{E}[Z]))] \leq \mathbb{E}\left[\exp \left(\lambda\left(Z-Z^{\prime}\right)\right]\right.
$$

现在，我们注意到一个奇怪的事实：差异$Z-Z'$关于零对称，因此如果$S \in\{-1,1\}$是一个随机符号变量，则$S\left(Z-Z^{\prime}\right)$完全和$Z-Z^{\prime}$同分布。因此我们可得：

$$
\begin{aligned} 
\mathbb{E}_{Z, Z^{\prime}}\left[\exp \left(\lambda\left(Z-Z^{\prime}\right)\right)\right] &=\mathbb{E}_{Z, Z^{\prime}, S}\left[\exp \left(\lambda S\left(Z-Z^{\prime}\right)\right)\right] \\ 
&=\mathbb{E}_{Z, Z^{\prime}}\left[\mathbb{E}_{S}\left[\exp \left(\lambda S\left(Z-Z^{\prime}\right)\right) | Z, Z^{\prime}\right]\right] 
\end{aligned}
$$

现在我们用不等式$(3)$对随机符号的矩母函数求导可以得到：

$$
\mathbb{E}_{S}\left[\exp \left(\lambda S\left(Z-Z^{\prime}\right)\right) | Z, Z^{\prime}\right] \leq \exp \left(\frac{\lambda^{2}\left(Z-Z^{\prime}\right)^{2}}{2}\right)
$$

当然，假设我们有$\left|Z-Z^{\prime}\right| \leq(b-a),$ so $\left(Z-Z^{\prime}\right)^{2} \leq(b-a)^{2}$，这就得到了：

$$
\mathbb{E}_{Z, Z^{\prime}}\left[\exp \left(\lambda\left(Z-Z^{\prime}\right)\right)\right] \leq \exp \left(\frac{\lambda^{2}(b-a)^{2}}{2}\right)
$$

这就是结果（除了因子$2$而不是$8$）。

现在我们用Hoeffding引理来证明定理$4$，只给出了上尾（即，$\frac{1}{n} \sum_{i=1}^{n}\left(Z_{i}-\mathbb{E}\left[Z_{i}\right]\right) \geq t$的概率）因为下面的尾巴也有类似的证明。我们使用切诺夫边界可以得到：

$$
\begin{aligned} 
\mathbb{P}\left(\frac{1}{n} \sum_{i=1}^{n}\left(Z_{i}-\mathbb{E}\left[Z_{i}\right]\right) \geq t\right) &=\mathbb{P}\left(\sum_{i=1}^{n}\left(Z_{i}-\mathbb{E}\left[Z_{i}\right]\right) \geq n t\right) \\ & \leq \mathbb{E}\left[\exp \left(\lambda \sum_{i=1}^{n}\left(Z_{i}-\mathbb{E}\left[Z_{i}\right]\right)\right)\right] e^{-\lambda n t} \\
&=\left(\prod_{i=1}^{n} \mathbb{E}\left[e^{\lambda\left(Z_{i}-\mathbb{E}\left[Z_{2}\right)\right.}\right]\right) e^{-\lambda n t} \begin{array}{l}{(i)} \\ { \leq}\end{array}\left(\prod_{i=1}^{n} e^{\frac{\lambda^{2}(b-\alpha)^{2}}{8}}\right) e^{-\lambda n t}
\end{aligned}
$$

不等式$(i)$在Hoeffding引理（引理$5$）中。稍微重写一下，然后在$\lambda\ge 0$上最小化，我们可得：

$$
\mathbb{P}\left(\frac{1}{n} \sum_{i=1}^{n}\left(Z_{i}-\mathbb{E}\left[Z_{i}\right]\right) \geq t\right) \leq \min _{\lambda \geq 0} \exp \left(\frac{n \lambda^{2}(b-a)^{2}}{8}-\lambda n t\right)=\exp \left(-\frac{2 n t^{2}}{(b-a)^{2}}\right)
$$

跟希望的结果一样。