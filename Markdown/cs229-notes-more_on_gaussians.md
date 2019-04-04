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


### 更多关于多元高斯分布

#### 介绍

到目前为止的课堂上，多元高斯以及出现在许多应用中，比如线性回归的概率解释、高斯判别分析、高斯混合聚类，以及最近的因子分析。在本节的笔记中，我们试图揭开多元高斯函数在最近的因子分析课程中引入的一些奇特的性质。本节笔记的目的是让大家对这些性质的来源有一些直观的了解，这样你就可以在作业（提醒你写作业的线索！）中更加明确地使用这些性质。

#### 1. 定义

我们称一个概率密度函数是一个均值为$\mu\in R^n$，协方差矩阵为$\Sigma\in S_{++}^n$的$^1$一个**多元正态分布（或高斯分布）(multivariate normal (or Gaussian) distribution)，** 其随机变量是向量值$x\in R^n$，该概率密度函数可以通过下式表达：

<blockquote><details><summary>上一小段上标1的说明（详情请点击本行）</summary>

1 复习一下线性代数章节中介绍的$S_{++}^n$是一个对称正定的$n\times n$矩阵空间，定义为：

$$
S_{++}^n=\{A\in R^{n\times n}:A=A^T\quad and\quad x^TAx>0\quad for\quad all\quad x\in R^n\quad such\quad that\quad x\neq 0\}
$$

</details></blockquote>

$$
p(x;\mu,\Sigma)=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))
$$

我们可以写作$x\sim\mathcal{N}(\mu,\Sigma)$

#### 2. 高斯分布的特点

多元高斯在实践中非常方便，原因如下:

- **特点 #1：** 如果你知道以$x$为随机变量的高斯分布的均值$\mu$和协方差矩阵$\Sigma$。则你可以直接写出关于$x$的概率密度函数。

- **特点 #2：** 下列高斯积分有闭式解(closed-form solutions)：

$$
\begin{aligned}
\int_{x\in R^n}p(x;\mu,\Sigma)dx &= \int_{-\infin}^{\infin}\dots\int_{-\infin}^{\infin}p(x;\mu,\Sigma)dx_1\dots dx_2=1 \\
\int_{x\in R^n}x_ip(x;\mu,\sigma)dx &= \mu_i \\
\int_{x\in R^n}(x_i-\mu_i)(x_j-\mu_j)p(x;\mu,\sigma)dx &=\Sigma_{ij}
\end{aligned}
$$

- **特点 #3：** 高斯函数遵循一些封闭性质(closure properties:)：
    - 独立高斯随机变量的和是高斯分布。
    - 联合高斯分布的边缘分布是高斯分布。
    - 联合高斯分布的条件是高斯分布。

乍一看，这些事实中的一些结论，尤其是第$1$和第$2$条，似乎要么是直观上显而易见的，要么至少是可信的。然而，我们可能不太清楚的是为什么这些特点如此有用。在本文档中，我们将提供一些直观解释说明如何在平常操作处理多元高斯随机变量时使用这些特点。

#### 3. 封闭性质

在本节中，我们将详细讨论前面描述的每个闭包属性，我们将使用事实#1和#2来证明属性，或者至少给出一些关于属性为真原因的直觉。

下面是我们本节将要介绍的内容的路线图：

||独立高斯分布的和|联合高斯分布的边缘分布|联合高斯分布的条件分布|
|:-:|:-:|:-:|:-:|
|为什么是高斯函数的解释|不介绍|介绍|介绍|
|结果的概率密度函数|介绍|介绍|介绍|

##### 3.1 独立高斯分布的和是高斯分布

本规则的正式表述为：

设有$y\sim\mathcal{N}(\mu,\Sigma)$和$z\sim\mathcal{N}(\mu',\Sigma')$为度量高斯分布的随机变量，其中$\mu,\mu'\in R^n$且$\Sigma,\Sigma'\in S_{++}^n$。则它们的和也同样是高斯分布：

$$
y+z\sim\mathcal{N}(\mu+\mu',\Sigma+\Sigma')
$$

在我们证明上面的结论前，先给出一些直观结果：

1. 首先要指出的是上述规则中独立假设的重要性。为了了解为什么这很重要，假设$y\sim\mathcal{N}(\mu,\sigma)$是服从于均值$\mu$方差$\sigma$的多元正态分布，并且假设$z=-y$。很明显，$z$也是服从于与多元高斯分布（事实上，$z\sim\mathcal{N}(-\mu,\sigma)$），但是$y+z$等于零！
2. 第二件需要指出的事情是许多学生感到困惑的一点：如果我们把两个高斯概率密度函数（多维空间中的“肿块(bumps)”）加在一起，我们会得到一些峰（即“双峰(two-humped)”的概率密度函数)么？在这里，我们要注意到随机变量$y + z$的概率密度函数并不是简单的将两个单独的概率密度函数的随机变量$y$和$z$相加，而是会变成$y$和$z$的卷积的概率密度函数。$^2$ 然而证明“两个高斯概率密度函数的卷积得到一个高斯概率密度函数”超出了这门课的范围。

<blockquote><details><summary>上一小段上标2的说明（详情请点击本行）</summary>

2 例如，如果$y$和$z$是单变量高斯函数（即：$y\sim\mathcal{N}(\mu,\sigma^2),z\sim\mathcal{N}(\mu,\sigma'^2)$），则它们的概率密度的卷积由下式给出：

$$
\begin{aligned}
p(y+z;\mu,\mu',\sigma,\sigma'^2) &=\int_{-\infin}^{\infin}p(w;\mu,\sigma^2)p(y+z-w;\mu',\sigma'^2)dw \\
&= \int_{-\infin}^{\infin}\frac 1{\sqrt{2\pi}\sigma}exp(-\frac 1{2\sigma^2}(w-\mu)^2)\cdot \frac 1{\sqrt{2\pi}\sigma'}exp(-\frac 1{2\sigma'^2}(y+z-w-\mu')^2)dw
\end{aligned}
$$

</details></blockquote>

转换一下思路，让我们用卷积给出高斯概率密度函数的观察结果，加上特点#1，来算出概率密度函数$p(y+z|\mu,\Sigma)$的解析解。如果我们要计算卷积。我们该怎么做呢？回顾特点#1，高斯分布完全由它的均值向量和协方差矩阵指定。如果我们能确定这些值是什么，那么我们就能计算出其解析解了。

这很简单！对应期望而言，我们有：

$$
E[y_i+z_i]=E[y_i]+E[z_i]=\mu_i+\mu_i'
$$

上式的结果根据期望的线性性质。因此，$y + z$的均值可以简单的写作$\mu+\mu'$。 同时，协方差矩阵的第$(i, j)$项由下式给出:

$$
\begin{aligned}
&E[(y_i+z_i)(y_j+z_j)]-E[y_i+z_i]E[y_j+z_j] \\
&\qquad=E[y_iy_j+z_iy_j+y_iz_j+z_iz_j]-(E[y_i]+E[z_i])(E[y_j]+E[z_j]) \\
&\qquad=E[y_iy_j]+E[z_iy_j]+E[y_iz_j]+E[z_iz_j]-E[y_i]E[y_j]-E[z_i]E[y_j]-E[y_i]E[z_j]-E[z_i]E[z_j] \\
&\qquad=(E[y_iy_j]-E[y_i]E[y_j])+(E[z_iz_j]-E[z_i]E[z_j]) \\
&\qquad\qquad+(E[z_iy_j]-E[z_i]E[y_j])+(E[y_iz_j]-E[y_i]E[z_j]) \\
\end{aligned}
$$

利用$y$和$z$相互独立的事实，我们得到$E[z_iy_j]=E[z_i]E[y_j]$和$E[y_iz_j]=E[y_i]E[z_j]$。因此，最后两项消去了，剩下：

$$
\begin{aligned}
&E[(y_i+z_i)(y_j+z_j)]-E[y_i+z_i]E[y_j+z_j] \\
&\qquad=(E[y_iy_j]-E[y_i]E[y_j])+(E[z_iz_j]-E[z_i]E[z_j]) \\
&\qquad=\Sigma_{ij}+\Sigma_{ij}'
\end{aligned}
$$

由此，我们可以得出$y + z$的协方差矩阵可以简单的写作$\Sigma+\Sigma'$。

此刻，让我们回顾一下刚刚我们做了什么？利用一些简单的期望和独立性的性质，我们计算出了$y + z$的均值和协方差矩阵。根据特点#1，我们可以立即写出$y + z$的概率密度函数，而不需要做卷积！$^3$

>3 当然，我们首先需要知道$y + z$是高斯分布。