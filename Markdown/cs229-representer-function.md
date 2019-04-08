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

