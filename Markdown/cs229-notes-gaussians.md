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

