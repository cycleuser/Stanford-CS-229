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


### 多元高斯分布

#### 介绍

我们称一个概率密度函数是一个均值为$\mu\in R^n$，协方差矩阵为$\Sigma\in S_{++}^n$$^1$的一个**多元正态分布（或高斯分布）(multivariate normal (or Gaussian) distribution)，** 其随机变量是向量值$X=[X_1\dots X_n]^T$，该概率密度函数$^2$可以通过下式表达：

<blockquote><details><summary>上一小段上标1,2的说明（详情请点击本行）</summary>

1 复习一下线性代数章节中介绍的$S_{++}^n$是一个对称正定的$n\times n$矩阵空间，定义为：

$$
S_{++}^n=\{A\in R^{n\times n}:A=A^T\quad and\quad x^TAx>0\quad for\quad all\quad x\in R^n\quad such\quad that\quad x\neq 0\}
$$

2 在我们的这部分笔记中，不使用$f_X(\bullet)$（如概率论注释一节所述），而是使用符号$p(\bullet)$代表概率密度函数。

</details></blockquote>

$$
p(x;\mu,\Sigma)=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))
$$

我们可以将其简写做$X\sim\mathcal{N}(\mu,\Sigma)$。在我们的这部分笔记中，我们描述了多元高斯函数及其一些基本性质。

#### 1. 与单变量高斯函数的关系

回忆一下，**一元正态分布（或高斯分布）(univariate normal (or Gaussian) distribution)** 的密度函数是由下式给出：

$$
p(x;\mu,\sigma^2)=\frac 1{\sqrt{2\pi}\sigma}exp(-\frac 1{2\sigma^2}(x-\mu)^2)
$$

这里，指数函数的自变量$-\frac 1{2\sigma^2}(x-\mu)^2$是关于变量$x$的二次函数。此外，抛物线是向下的，因为二次项的系数是负的。前面的系数$\frac 1{\sqrt{2\pi}\sigma}$是不依赖$x$的常数。因此，我们可以简单地把这个系数当作保证下面的式子成立的“标准化因子”(normalization factor)。

$$
\frac 1{\sqrt{2\pi}\sigma}\int_{-\infin}^{\infin} exp(-\frac 1{2\sigma^2}(x-\mu)^2)=1
$$

![](https://github.com/Kivy-CN/Stanford-CS-229-CN/blob/master/img/cs229notegf1.png?raw=true)

在多元高斯概率密度函数的情况下，指数函数的自变量$-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)$是一个以向量形式的$x$为变量的**二次形(quadratic form)**。因为$\Sigma$是正定矩阵，并且任何正定矩阵的逆也是正定的，那么对于任何非零向量$z$，有$z\Sigma^Tz>0$。这就暗示了任何向量$x\neq\mu$，有：

$$
(x-\mu)^T\Sigma^{-1}(x-\mu)>0 \\
-\frac 12(x-\mu)^T\Sigma^{-1}(x-\mu)<0
$$

就像在单变量的情况下，你可以把指数函数的参数看成是一个开口向下的二次曲线。指数函数前面的系数（即，$\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}$）是一个比单变量情况下更复杂的一种形式。但是，它仍然不依赖于$x$，因此它只是一个用来保证下面的式子成立的标准化因子：

$$
\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}\int_{-\infin}^{\infin}\int_{-\infin}^{\infin}\dots\int_{-\infin}^{\infin}exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))dx_1dx_2\dots dx_n=1
$$

#### 2. 协方差矩阵

**协方差矩阵**的概念对于理解多元高斯分布是至关重要的。回忆一下，对于一对随机变量$X$和$Y$，它们的协方差定义为：

$$
Cov[X,Y]=E[(X-E[X])(Y-E[Y])]=E[XY]-E[X]E[Y]
$$

当处理多个变量时，协方差矩阵提供了一种简洁的方法来总结所有对变量的协方差。特别注意协方差矩阵，我们通常将其表示成一个$n\times n$的矩阵$\Sigma$，其中第$(i,j)$个元素代表$Cov[X_i,Y_j]$

下面的命题（其证明见附录A.1）给出了描述随机向量$X$的协方差矩阵的另一种方法：

**命题 1.** 对于任意一个均值为$\mu$，随机向量为$X$的协方差矩阵$\Sigma$：

$$
\Sigma=E[(X-\mu)(X-\mu)^T]=E[XX^T]-\mu\mu^T
$$

在多元高斯的定义中，我们要求协方差矩阵$\Sigma$是对称正定矩阵（即，$\Sigma\in S_{++}^n$）。为什么存在这种限制？如下式所示，任意随机向量的协方差矩阵都必须是对称正半定的：

**命题 2.** 假如$\Sigma$是关于随机向量$X$的协方差矩阵。则$\Sigma$是对称半正定矩阵。

证明。$\Sigma$的对称性直接来源于它的定义。之后，对于任意向量$z\in R^n$我们可以观察到：

$$
\begin{aligned}
z^T\Sigma z &= \sum_{i=1}^n\sum_{j=1}^n(\Sigma_{ij}z_iz_j)\qquad\qquad &(2) \\
&= \sum_{i=1}^n\sum_{j=1}^n(Cov[X_i,X_j]\cdot z_iz_j)   \\
&= \sum_{i=1}^n\sum_{j=1}^n(E[(X_i-E[X_i])(X_j-E[X_j])] \cdot z_iz_j)  \\
&= E[\sum_{i=1}^n\sum_{j=1}^n(X_i-E[X_i])(X_j-E[X_j])\cdot z_iz_j]&(3)
\end{aligned}
$$

这里，$(2)$由二次形式的展开公式（参见线性代数部分章节）得到，$(3)$由期望的线性性质得到（参见概率章节）。

要完成证明，请注意括号内的数量是形式$\sum_{i=1}^n\sum_{j=1}^nx_ix_jz_iz_j=(x^Tz)^2\ge 0$（见问题设定#1）。因此，期望中的量总是非负的，即得到期望本身必须是非负的。我们可以断定$z^T\Sigma z\ge 0$

从上面的命题可以推出，为了使$\Sigma$成为一个有效的协方差矩阵，其必须是对称正半定的。然而，为了使$\Sigma^{-1}$存在（如多元高斯密度的定义所要求的），则$\Sigma$必须是可逆的，因此是满秩的。由于任何满秩对称正半定矩阵必然是对称正定的，因此$\Sigma$必然是对称正定的。

####3. 对角协方差矩阵的情况

为了直观地理解多元高斯函数是什么，考虑一个简单的例子 并且协方差矩阵 是对角阵，即：

$$
x=\begin{bmatrix}x_1\\x_2\end{bmatrix}\qquad\qquad \mu=\begin{bmatrix}\mu_1\\\mu_2\end{bmatrix}\qquad\qquad \Sigma=\begin{bmatrix}\sigma_1^2&0\\0&\sigma_2^2\end{bmatrix}
$$

在这种情况下，多元高斯密度的形式是：

$$
\begin{aligned}
p(x;\mu,\Sigma) &=\frac{1}{2\pi\begin{vmatrix}\sigma_1^2&0\\0&\sigma_2^2\end{vmatrix}^{1/2}} exp(-\frac{1}{2}\begin{bmatrix}x_1-\mu_1\\x_2-\mu_2\end{bmatrix}^T\begin{bmatrix}\sigma_1^2&0\\0&\sigma_2^2\end{bmatrix}^{-1}\begin{bmatrix}x_1-\mu_1\\x_2-\mu_2\end{bmatrix}) \\
&= \frac 1{2\pi(\sigma_1^2\cdot \sigma_2^2-0\cdot 0)^{1/2}}exp(-\frac{1}{2}\begin{bmatrix}x_1-\mu_1\\x_2-\mu_2\end{bmatrix}^T\begin{bmatrix}\frac 1{\sigma_1^2}&0\\0&\frac 1{\sigma_2^2}\end{bmatrix}^{-1}\begin{bmatrix}x_1-\mu_1\\x_2-\mu_2\end{bmatrix})
\end{aligned}
$$

这里我们依赖于一个$2\times 2$矩阵$^3$的行列式的显式公式，事实上一个对角矩阵的逆就是通过取每个对角元素的倒数来找到的。之后可得：

>3 即$\begin{vmatrix}a&b\\c&d\end{vmatrix}=ad-bc$

$$
\begin{aligned}
p(x;\mu,\Sigma) &=\frac{1}{2\pi\sigma_1\sigma_2} exp(-\frac{1}{2}\begin{bmatrix}x_1-\mu_1\\x_2-\mu_2\end{bmatrix}^T\begin{bmatrix}\frac 1{\sigma_1^2}(x_1-\mu_1)\\\frac 1{\sigma_2^2}(x_2-\mu_2)\end{bmatrix}) \\
&= \frac{1}{2\pi\sigma_1\sigma_2} exp(-\frac 1{2\sigma_1^2}(x_1-\mu_1)^2-\frac 1{2\sigma_2^2}(x_2-\mu_2)^2) \\
&= \frac{1}{\sqrt{2\pi}\sigma_1} exp(-\frac 1{2\sigma_1^2}(x_1-\mu_1)^2)\cdot \frac{1}{\sqrt{2\pi}\sigma_2} exp(-\frac 1{2\sigma_2^2}(x_2-\mu_2)^2)
\end{aligned}
$$

最后一个方程是两个独立的高斯登函数的乘积，其中一个具有均值$\mu_1$，方差$\sigma_1^2$。另一个具有均值$\mu_2$，方差$\sigma_2^2$。

更一般地，我们可以证明$n$维高斯函数的均值$\mu\in R^n$并且对角协方差矩阵$\sigma=diag(\sigma_1^2,\sigma_2^2,\dots,\sigma_n^2)$与$n$个独立的均值和方差分别是$\mu_i,\sigma_i^2$的凹高斯随机变量的集合相同。

#### 4. 等高线

从概念上理解多元高斯函数的另一种方法是理解其**等高线**的形状。对于一个函数$f:R^2\rightarrow R$，等高线数学形状的集合定义如下：

$$
\{x\in R^2:f(x)=c\}
$$

其中$c\in R$。$^4$

>4 等值线通常也称为**等值线(level curves)。** 更一般地说，函数的一组**水平集(level set)** $f:R^2\rightarrow R$，其实一个对于一些$c\in R$形式为$\{x\in R^2:f(x)=c\}$的集合。

##### 4.1 等高线的型状

多元高斯函数的等值线是什么样的？和之前一样，我们考虑$n = 2$的情况，

$$
x=\begin{bmatrix}x_1\\x_2\end{bmatrix}\qquad\qquad \mu=\begin{bmatrix}\mu_1\\\mu_2\end{bmatrix}\qquad\qquad \Sigma=\begin{bmatrix}\sigma_1^2&0\\0&\sigma_2^2\end{bmatrix}
$$

正如我们在上一节所展示的，

$$
p(x;\mu,\Sigma) = \frac{1}{2\pi\sigma_1\sigma_2} exp(-\frac 1{2\sigma_1^2}(x_1-\mu_1)^2-\frac 1{2\sigma_2^2}(x_2-\mu_2)^2)\qquad\qquad(4)
$$

现在，让我们考虑由所有点组成的水平集，其中对于某个常数$c\in R$来说$p(x;\mu,\sigma)=c$。 特别的，考虑对于所有$x_1,x_2\in R$的集合，比如：

$$
\begin{aligned}
c&=\frac{1}{2\pi\sigma_1\sigma_2} exp(-\frac 1{2\sigma_1^2}(x_1-\mu_1)^2-\frac 1{2\sigma_2^2}(x_2-\mu_2)^2) \\
2\pi c\sigma_1\sigma_2 &= exp(-\frac 1{2\sigma_1^2}(x_1-\mu_1)^2-\frac 1{2\sigma_2^2}(x_2-\mu_2)^2)  \\
log(2\pi c\sigma_1\sigma_2) &= -\frac 1{2\sigma_1^2}(x_1-\mu_1)^2-\frac 1{2\sigma_2^2}(x_2-\mu_2)^2 \\
log(\frac 1{2\pi c\sigma_1\sigma_2}) &= \frac 1{2\sigma_1^2}(x_1-\mu_1)^2+\frac 1{2\sigma_2^2}(x_2-\mu_2)^2 \\
1 &= \frac {(x_1-\mu_1)^2}{2\sigma_1^2log(\frac 1{2\pi c\sigma_1\sigma_2})}+\frac {(x_2-\mu_2)^2}{2\sigma_2^2log(\frac 1{2\pi c\sigma_1\sigma_2})}
\end{aligned}
$$

定义：

$$
r_1= \sqrt{2\sigma_1^2log(\frac 1{2\pi c\sigma_1\sigma_2})}\qquad\qquad r_2= \sqrt{2\sigma_2^2log(\frac 1{2\pi c\sigma_1\sigma_2})}
$$

之后可得：

$$
1 = (\frac {x_1-\mu_1}{r_1})^2+(\frac {x_2-\mu_2}{r_2})^2\qquad\qquad (5)
$$

方程$(5)$在高中解析几何中应该很熟悉：它是一个**轴向椭圆(axis-aligned ellipse)** 的方程，其中心是$(\mu_1,\mu_2)$，并且$x_1$轴的长度是$2r_1$，$x_2$轴的长度是$2r_2$。

![](https://github.com/Kivy-CN/Stanford-CS-229-CN/blob/master/img/cs229notegf2.png?raw=true)
左边的图显示了一个热图，它表示具有均值为$\mu=\begin{bmatrix}3\\2\end{bmatrix}$，对角协方差矩阵为$\Sigma=\begin{bmatrix}25&0\\0&9\end{bmatrix}$的轴向多元高斯函数的概率密度函数值。注意到这个高斯分布的中心点为$(3,2)$，等高线均为椭圆形，长/短轴长之比为$5:3$。右边的图显示了一个热图，该图表示了一个非轴向对齐的具有平均值为$\mu=\begin{bmatrix}3\\2\end{bmatrix}$协方差矩阵为$\Sigma=\begin{bmatrix}25&5\\5&5\end{bmatrix}$的多元高斯概率密度函数值。这里，椭圆再次以$(3,2)$为中心，但现在通过线性变换旋转了主轴和副主轴。