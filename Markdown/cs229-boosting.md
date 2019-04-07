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


### 提升(Boosting)

#### 1. 提升

到目前为止，我们已经了解了如何在已经选择了数据表示的情况下解决分类（和其他）问题。我们现在讨论一个称为boost的过程，它最初是由Rob Schapire发现的，后来由Schapire和Yoav Freund进一步开发，它自动选择特性表示。我们采用了一种基于优化的视角，这与Freund和Schapire最初的解释和论证有些不同，但这有助于我们的方法：$(1)$选择模型表达，$(2)$选择损失，$(3)$最小化损失函数。

在阐明这个问题之前，我们先对我们要做的事情有一点直觉。粗略地说，提升的思想是用一个弱学习算法——任何一个学习算法只要给出一个比随机好一点的分类器——然后把它转换成一个比随机好很多的强分类器。为了对此建立一些直觉，考虑一个的数字识别案例，在这个案例中，我们希望区分$0$和$1$，我们接收到的图像必须进行分类。那么一个自然的弱学习器可能会取图像的中间像素，如果它是彩色的，就把图像称为$1$，如果它是空白的，就把图像称为$0$。这个分类器可能远远不够完美，但它可能比随机分类器要好。提升过程通过收集这些弱分类器，然后对它们的贡献进行加权，从而形成一个比任何单个分类器精度都高得多的分类器。

考虑到这一点，让我们来阐明这个问题。我们对提升的理解是在无限维空间中采用坐标下降法，虽然听起来很复杂，但并不像看上去那么难。首先，假设我们有原始标签为$y \in\{-1,1\}$的输入示例$x \in \mathbb{R}^{n}$，该示例在二分类中很常见。我们还假设我们有无穷多个特征函数$\phi_{j} : \mathbb{R}^{n} \rightarrow\{-1,1\}$以及无限向量$\theta=\left[ \begin{array}{lll}{\theta_{1}} & {\theta_{2}} & {\cdots}\end{array}\right]^{T}$，然而我们总假设只有有限数量的非零元素。我们使用的分类器如下：

$$
h_{\theta}(x)=\operatorname{sign}\left(\sum_{j=1}^{\infty} \theta_{j} \phi_{j}(x)\right)
$$

这里让我们随便使用下符号，并定义$\theta^{T} \phi(x)=\sum_{j=1}^{\infty} \theta_{j} \phi_{j}(x)$。

在提升中，我们通常称特性$\phi_{j}$为弱假设。给定一个训练集$\left(x^{(1)}, y^{(1)}\right), \ldots,\left(x^{(m)}, y^{(m)}\right)$，我们称一个向量$p=\left(p^{(1)}, \ldots, p^{(m)}\right)$为样本中的分布，如果对于所有$i$以及下式成立时有$p^{(i)} \geq 0$。

$$
\sum_{i=1}^{m} p^{(i)}=1
$$

然后我们称一个弱学习器具有边界$\gamma>0$，如果对于任意在第$m$个训练样本的分布$p$存在一个弱假设$\phi_j$，使得：

$$
\sum_{i=1}^{m} p^{(i)} 1\left\{y^{(i)} \neq \phi_{j}\left(x^{(i)}\right)\right\} \leq \frac{1}{2}-\gamma \qquad\qquad(1)
$$

也就是说，我们假设有一些分类器比对数据集的随机猜测稍微好一些。弱学习算法的存在只是一个假设，但令人惊讶的是，我们可以将任何弱学习算法转换成一个具有精度完美的算法。

在更一般的情况下，我们假设我们访问了一个弱学习器，这是一个算法，它以训练示例上的一个分布（权重）$p$作为输入，并返回一个性能略好于随机分类器的分类器。我们将展示了在给定弱学习算法的条件下，boost如何返回训练数据具有完美精度的分类器。（诚然，我们希望分类器能够很好地推广到不可见的数据，但目前我们忽略了这个问题。)


