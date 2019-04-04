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

