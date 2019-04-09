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

让我们考虑使得等式$(2)$中的风险最小化而选择的$\theta$的结构。正如我们通常做的那样，我们假设对于每一个固定目标值$y \in \mathcal{Y}$，函数$\mathrm{L}(z, y)$是关于$z$的凸函数。（这是线性回归、二元和多级逻辑回归的情况，以及我们将考虑的其他一些损失。）结果表明，在这些假设下，我们总是可以把问题$(2)$的解写成输入变量$x^{(i)}$的线性组合。更准确地说，我们有以下定理，称为表示中心定理。

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