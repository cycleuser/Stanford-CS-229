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

##### 1.1 提升算法

粗略地说，boost首先在数据集中为每个训练示例分配相等的权重。然后，它接收到一个弱假设，该弱假设根据当前训练实例的权重表现良好，并将其合并到当前的分类模型中。然后对训练实例进行权重调整，使出错的例子获得更高的权重，使弱学习算法集中于分类器对这些例子的处理，而没有错误的例子获得更低的权重。这种训练数据的重复权重调整，加上一个学习能力较差的学习者在分类器当前表现不佳的示例上做得很好，就产生了性能良好的分类器。

该算法对分类问题的指数损失进行坐标下降，目标为：

$$
J(\theta)=\frac{1}{m} \sum_{i=1}^{m} \exp \left(-y^{(i)} \theta^{T} \phi\left(x^{(i)}\right)\right)
$$

我们首先展示如何计算损失$J(\theta)$的坐标下降更新的确切形式，坐标下降迭代如下：

(i) 选择一个坐标$j \in \mathbb{N}$

(ii) 更新$\theta_j$为：

$$
\theta_{j}=\underset{\theta_{j}}{\operatorname{argmin}} J(\theta)
$$

对于所有$k\neq j$不相同时留下$\theta_k$。

我们迭代上面的过程直到收敛。

在提升的情况下，由于自然指数函数分析很方便，坐标更新的推导并不困难。现在我们展示如何进行更新。假设我们想要更新坐标$k$。定义：

$$
w^{(i)}=\exp \left(-y^{(i)} \sum_{j \neq k} \theta_{j} \phi_{j}\left(x^{(i)}\right)\right)
$$

为权重，并注意优化坐标$k$达到相应的最小化：

$$
\sum_{i=1}^{m} w^{(i)} \exp \left(-y^{(i)} \phi_{k}\left(x^{(i)}\right) \alpha\right)
$$

在$\alpha=\theta_k$时。现在定义：

$$
W^{+} :=\sum_{i : y^{(i)} \phi_{k}\left(x^{(i)}\right)=1} w^{(i)} \text { and } W^{-} :=\sum_{i : y^{(i)}\phi_{k}\left(x^{(i)}\right)=-1} w^{(i)}
$$

为了实例权值之和使得$\phi_k$分别分类正确和错误。找出$\theta_k$与下面的式子的选择相同：

$$
\alpha=\underset{\alpha}{\arg \min }\left\{W^{+} e^{-\alpha}+W^{-} e^{\alpha}\right\}=\frac{1}{2} \log \frac{W^{+}}{W^{-}}
$$

为了得到最终的等式，我们求导并将其设为零。我们得到$-W^{+} e^{-\alpha}+W^{-} e^{\alpha}=0$。即$W^{-} e^{2 \alpha}=W^{+}$或$\alpha=\frac{1}{2} \log \frac{W^{+}}{W^{-}}$。

剩下的就是选择特定的坐标来执行坐标下降。我们假设我们已经使用了如图$1$所示的弱学习器算法，在迭代$t$分布作为输入$p$的训练集，并返回一个弱假设$\phi_t$满足边界条件$(1)$。我们提出总提升算法如图$2$所示。它在迭代$t=1,2,3, \ldots$中进行。我们表示弱学习算法返回的假设集在$t$时刻产生$\left\{\phi_{1}, \dots, \phi_{t}\right\}$。

![](https://raw.githubusercontent.com/Kivy-CN/Stanford-CS-229-CN/master/img/cs229notebf1.png)

![](https://raw.githubusercontent.com/Kivy-CN/Stanford-CS-229-CN/master/img/cs229notebf1.png)

#### 2 提升的收敛性

我们现在认为，提升程序达到$0$训练误差，我们也提供了收敛速度为零。为此，我们提出了一个保证取得提升的引理。

**引理 2.1** 令：

$$
J\left(\theta^{(t)}\right)=\frac{1}{m} \sum_{i=1}^{m} \exp \left(-y^{(i)} \sum_{\tau=1}^{t} \theta_{\tau} \phi_{\tau}\left(x^{(i)}\right)\right)
$$

则：

$$
J\left(\theta^{(t)}\right) \leq \sqrt{1-4 \gamma^{2}} J\left(\theta^{(t-1)}\right)
$$

由于引理的证明是有点复杂的，而不是本节笔记的中心——尽管知道该算法会收敛是很重要的！——我们把证明推迟到附录A.1。让我们描述一下它如何保证增强过程收敛到一个训练误差为零的分类器。

我们在$\theta^{(0)}=\overrightarrow{0}$处初始化过程，以便初始经验损失为$J\left(\theta^{(0)}\right)=1$。现在,我们注意到对于任何$\theta$，误分类误差满足：

$$
1\left\{\operatorname{sign}\left(\theta^{T} \phi(x)\right) \neq y\right\}=1\left\{y \theta^{T} \phi(x) \leq 0\right\} \leq \exp \left(-y \theta^{T} \phi(x)\right)
$$

因为对于所有$z\ge 0$有$e^z\ge 1$。由此可知，误分类错误率具有上界：

$$
\frac{1}{m} \sum_{i=1}^{m} 1\left\{\operatorname{sign}\left(\theta^{T} \phi\left(x^{(i)}\right)\right) \neq y^{(i)}\right\} \leq J(\theta)
$$

因此如果$J(\theta)<\frac{1}{m}$，则向量$\theta$在训练样本中没有错误。经过$t$次提升迭代，得到经验风险满足：

$$
J\left(\theta^{(t)}\right) \leq\left(1-4 \gamma^{2}\right)^{\frac{t}{2}} J\left(\theta^{(0)}\right)=\left(1-4 \gamma^{2}\right)^{\frac{t}{2}}
$$

查找得出多少次迭代才能保证$J\left(\theta^{(t)}\right)<\frac{1}{m}$。我们对$J\left(\theta^{(t)}\right)<1 / m$取对数：

$$
\frac{t}{2} \log \left(1-4 \gamma^{2}\right)<\log \frac{1}{m}, \quad \text { or } \quad t>\frac{2 \log m}{-\log \left(1-4 \gamma^{2}\right)}
$$

用一阶泰勒展开，得到$\log \left(1-4 \gamma^{2}\right) \leq-4 \gamma^{2}$，我们看到，如果我们使用提升的轮数——我们使用弱分类器的数量——满足下面的条件：

$$
t>\frac{\log m}{2 \gamma^{2}} \geq \frac{2 \log m}{-\log \left(1-4 \gamma^{2}\right)}
$$

则$J\left(\theta^{(t)}\right)<\frac{1}{m}$。

#### 3 实现弱学习器(Implementing weak-learners)

增强算法的一个主要优点是，它们可以自动从原始数据中为我们生成特性。此外，由于弱假设总是返回$\{1,1\}$中的值，所以在使用学习算法时不需要将特征标准化，使其具有相似的尺度，这在实践中会产生很大的差异。此外。虽然这不是理论上易于理解，许多类型的弱学习器程序中引入非线性智能分类器，可以产生比到目前为止我们已经看到的更多的表达模型的简单线性模型形式$\theta^Tx$。