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


### 表示函数

#### 1. 广义损失函数

基于我们对监督学习的理解可知学习步骤：

- $(1)$选择问题的表示形式，
- $(2)$选择损失函数，
- $(3)$最小化损失函数。

让我们考虑一个稍微通用一点的监督学习公式。在我们已经考虑过的监督学习设置中，我们输入数据$x \in \mathbb{R}^{n}$和目标$y$来自空间$\mathcal{Y}$。在线性回归中，相应的$y \in \mathbb{R}$，即$\mathcal{Y}=\mathbb{R}$。在logistic回归等二元分类问题中，我们有$y \in \mathcal{Y}=\{-1,1\}$，对于多标签分类问题，我们对于分类数为$k$的问题有$y \in \mathcal{Y}=\{1,2, \ldots, k\}$。

对于这些问题，我们对于某些向量$\theta$基于$\theta^Tx$做了预测，我们构建了一个损失函数$\mathrm{L} : \mathbb{R} \times \mathcal{Y} \rightarrow \mathbb{R}$，其中$\mathrm{L}\left(\theta^{T} x, y\right)$用于测量我们预测$\theta^Tx$时的损失，对于logistic回归，我们使用logistic损失函数：

$$
\mathrm{L}(z, y)=\log \left(1+e^{-y z}\right) \text { or } \mathrm{L}\left(\theta^{T} x, y\right)=\log \left(1+e^{-y \theta^{T} x}\right)
$$

对于线性回归，我们使用平方误差损失函数：

$$
\mathrm{L}(z, y)=\frac{1}{2}(z-y)^{2} \quad \text { or } \quad \mathrm{L}\left(\theta^{T} x, y\right)=\frac{1}{2}\left(\theta^{T} x-y\right)^{2}
$$

对于多类分类，我们有一个小的变体，其中我们对于$\theta_{i} \in \mathbb{R}^{n}$来说令$\Theta=\left[\theta_{1} \cdots \theta_{k}\right]$，并且使用损失函数$\mathrm{L} : \mathbb{R}^{k} \times\{1, \ldots, k\} \rightarrow \mathbb{R}$：

$$
\mathrm{L}(z, y)=\log \left(\sum_{i=1}^{k} \exp \left(z_{i}-z_{y}\right)\right) \operatorname{or} \mathrm{L}\left(\Theta^{T} x, y\right)=\log \left(\sum_{i=1}^{k} \exp \left(x^{T}\left(\theta_{i}-\theta_{y}\right)\right)\right)
$$

这个想法是我们想要对于所有$i \neq k$得到$\theta_{y}^{T} x>\theta_{i}^{T}$。给定训练集合对$\left\{x^{(i)}, y^{(i)}\right\}$，通过最小化下面式子的经验风险来选择$\theta$：

$$
J(\theta)=\frac{1}{m} \sum_{i=1}^{m} L\left(\theta^{T} x^{(i)}, y^{(i)}\right)\qquad\qquad(1)
$$

#### 2 表示定理

让我们考虑一个稍微不同方法来选择$\theta$使得等式$(1)$中的风险最小化。在许多情况下——出于我们将在以后的课程中学习更多的原因——将正则化添加到风险$J$中是很有用的。我们添加正则化的原因很多：通常它使问题$(1)$容易计算出数值解，它还可以让我们使得等式$(1)$中的风险最小化而选择的$\theta$够推广到未知数据。通常，正则化被认为是形式$r(\theta)=\|\theta\|$ 或者 $r(\theta)=\|\theta\|^{2}$其中$\|\cdot\|$是$\mathbb{R}^{n}$中的范数。最常用的正则化是$l_2$-正则化，公式如下：

$$
r(\theta)=\frac{\lambda}{2}\|\theta\|_{2}^{2}
$$

其中$\|\theta\|_{2}=\sqrt{\theta^{T} \theta}$被称作向量$\theta$的欧几里得范数或长度。基于此可以得到正规化风险：

$$
J_{\lambda}(\theta)=\frac{1}{m} \sum_{i=1}^{m} \mathrm{L}\left(\theta^{T} x^{(i)}, y^{(i)}\right)+\frac{\lambda}{2}\|\theta\|_{2}^{2}\qquad\qquad(2)
$$

让我们考虑使得等式$(2)$中的风险最小化而选择的$\theta$的结构。正如我们通常做的那样，我们假设对于每一个固定目标值$y \in \mathcal{Y}$，函数$\mathrm{L}(z, y)$是关于$z$的凸函数。（这是线性回归、二元和多级逻辑回归的情况，以及我们将考虑的其他一些损失。）结果表明，在这些假设下，我们总是可以把问题$(2)$的解写成输入变量$x^{(i)}$的线性组合。更准确地说，我们有以下定理，称为表示定理。

**定理2.1** 假设在正规化风险$(2)$的定义中有$\lambda\ge 0$。然后，可以得到令正则化化风险$(2)$最小化的式子：

$$
\theta=\sum_{i=1}^{m} \alpha_{i} x^{(i)}
$$

其中$\alpha_i$为一些实值权重。

**证明** 为了直观，我们给出了在把$\mathrm{L}(x,y)$看做关于$z$的可微函数，并且$\lambda>0$的情况下结果的证明。详情见附录A，我们给出了定理的一个更一般的表述以及一个严格的证明。

令$\mathrm{L}^{\prime}(z, y)=\frac{\partial}{\partial z} L(z, y)$代表损失函数关于$z$的导致。则根据链式法则，我们得到了梯度恒等式：

$$
\nabla_{\theta} \mathrm{L}\left(\theta^{T} x, y\right)=\mathrm{L}^{\prime}\left(\theta^{T} x, y\right) x \text { and } \nabla_{\theta} \frac{1}{2}\|\theta\|_{2}^{2}=\theta
$$

其中$\nabla_{\theta}$代表关于$\theta$的梯度。由于风险在所有固定点（包括最小值点）的梯度必须为$0$，我们可以这样写：

$$
\nabla J_{\lambda}(\theta)=\frac{1}{m} \sum_{i=1}^{m} \mathrm{L}^{\prime}\left(\theta^{T} x^{(i)}, y^{(i)}\right) x^{(i)}+\lambda \theta=\overrightarrow{0}
$$

特别的，令$w_{i}=\mathrm{L}^{\prime}\left(\theta^{T} x^{(i)}, y^{(i)}\right)$，因为$\mathrm{L}^{\prime}\left(\theta^{T} x^{(i)}, y^{(i)}\right)$是一个标量（依赖于$\theta$，但是无论$\theta$是多少，$w_i$始终是一个实数），所以我们有：

$$
\theta=-\frac{1}{\lambda} \sum_{i=1}^{n} w_{i} x^{(i)}
$$

设$\alpha_{i}=-\frac{w_{i}}{\lambda}$以得到结果。

#### 3 非线性特征与核

基于表示定理$2.1$我们看到，我们可以写出向量$\theta$的作为数据$\left\{x^{(i)}\right\}_{i=1}^{m}$的线性组合。重要的是，这意味着我们总能做出预测：

$$
\theta^{T} x=x^{T} \theta=\sum_{i=1}^{m} \alpha_{i} x^{T} x^{(i)}
$$

也就是说，在任何学习算法中，我们都可以将$\theta^{T} x$替换成$\sum_{i=1}^{m} \alpha_{i} x^{(i)^{T}}x$，然后直接通过$\alpha \in \mathbb{R}^{m}$使其最小化。

让我们从更普遍的角度来考虑这个问题。在我们讨论线性回归时，我们遇到一个问题，输入$x$是房子的居住面积，我们考虑使用特征$x$，$x^2$和$x^3$（比方说）来进行回归，得到一个三次函数。为了区分这两组变量，我们将“原始”输入值称为问题的输入**属性**（在本例中，$x$是居住面积）。当它被映射到一些新的量集，然后传递给学习算法时，我们将这些新的量称为输入**特征**。（不幸的是，不同的作者使用不同的术语来描述这两件事，但是我们将在本节的笔记中始终如一地使用这个术语。）我们还将让$\phi$表示特征映射，映射属性的功能。例如，在我们的例子中，我们有：

$$
\phi(x)=\left[ \begin{array}{c}{x} \\ {x^{2}} \\ {x^{3}}\end{array}\right]
$$

与其使用原始输入属性$x$应用学习算法，不如使用一些特征$\phi(x)$来学习。要做到这一点，我们只需要回顾一下之前的算法，并将其中的$x$替换为$\phi(x)$。 

因为算法可以完全用内积$\langle x, z\rangle$来表示，这意味着我们可以将这些内积替换为$\langle\phi(x), \phi(z)\rangle$。特别是给定一个特征映射$\phi$，我们可以将相应**核**定义为：

$$
K(x, z)=\phi(x)^{T} \phi(z)
$$

然后，在我们之前的算法中，只要有$\langle x, z\rangle$，我们就可以用$K(x, z)$替换它，并且现在我们的算法可以使用特征$\phi$来学习。让我们更仔细地写出来。我们通过表示定理（定理2.1）看到我们可以对于一些权重$\alpha_i$写出$\theta=\sum_{i=1}^{m} \alpha_{i} \phi\left(x^{(i)}\right)$。然后我们可以写出（正则化）风险：

$$
\begin{aligned} 
J_{\lambda}(\theta) 
&=J_{\lambda}(\alpha) \\ 
&=\frac{1}{m} \sum_{i=1}^{m} L\left(\phi\left(x^{(i)}\right)^{T} \sum_{j=1}^{m} \alpha_{j} \phi\left(x^{(j)}\right), y^{(i)}\right)+\frac{\lambda}{2}\left\|\sum_{i=1}^{m} \alpha_{i} \phi\left(x^{(i)}\right)\right\|_{2}^{2} \\ 
&=\frac{1}{m} \sum_{i=1}^{m} L\left(\sum_{j=1}^{m} \alpha_{j} \phi\left(x^{(i)}\right)^{T} \phi\left(x^{(j)}\right), y^{(i)}\right)+\frac{\lambda}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} \phi\left(x^{(i)}\right)^{T} \phi\left(x^{(j)}\right) \\ 
&=\frac{1}{m} \sum_{i=1}^{m} \mathrm{L}\left(\sum_{j=1}^{m} \alpha_{j} K\left(x^{(i)}, x^{(j)}\right)+\frac{\lambda}{2} \sum_{i, j} \alpha_{i} \alpha_{i} K\left(x^{(i)}, x^{(j)}\right)\right.
\end{aligned}
$$

也就是说，我们可以把整个损失函数写成核矩阵的最小值：

$$
K=\left[K\left(x^{(i)}, x^{(j)}\right)\right]_{i, j=1}^{m} \in \mathbb{R}^{m \times m}
$$

现在，给定$\phi$，我们可以很容易地通过$\phi(x)$和$\phi(z)$和内积计算$K(x,z)$。但更有趣的是，通常$K(x,z)$可能非常廉价的计算，即使$\phi(x)$本身可能是非常难计算的（可能因为这是一个极高的维向量）。在这样的设置中，通过在我们的算法中一个有效的方法来计算$K(x,z)$，我们可以学习的高维特征空间空间由$\phi$给出，但没有明确的找到或表达向量$\phi(x)$。例如，一些核（对应于无限维的向量$\phi$）包括：

$$
K(x, z)=\exp \left(-\frac{1}{2 \tau^{2}}\|x-z\|_{2}^{2}\right)
$$

称为高斯或径向基函数(RBF)核，适用于任何维数的数据，或最小核（适用于$x\in R$）由下式得：

$$
K(x, z)=\min \{x, z\}
$$

有关这些内核机器的更多信息，请参见[支持向量机(SVMs)的课堂笔记](https://kivy-cn.github.io/Stanford-CS-229-CN/#/Markdown/cs229-notes3)。