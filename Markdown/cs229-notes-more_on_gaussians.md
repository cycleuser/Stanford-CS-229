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

到目前为止的课堂上，多元高斯分布已经出现在许多应用中，比如线性回归的概率解释、高斯判别分析、高斯混合聚类，以及最近学习的因子分析。在本节的笔记中，我们试图揭开多元高斯函数在最近学习的因子分析课程中引入的一些奇特的性质。本节笔记的目的是让大家对这些性质的来源有一些直观的了解，这样你就可以在作业（提醒你写作业的线索！）中更加明确地使用这些性质。

#### 1. 定义

我们称一个概率密度函数是一个均值为$\mu\in R^n$，协方差矩阵为$\Sigma\in S_{++}^n$的$^1$一个**多元正态分布（或高斯分布）(multivariate normal (or Gaussian) distribution)，** 其随机变量是向量值$x\in R^n$，该概率密度函数可以通过下式表达：

<blockquote><details><summary>上一小段上标1的说明（详情请点击本行）</summary>

1 复习一下线性代数章节中介绍的$S_{++}^n$是一个对称正定的$n\times n$矩阵空间，定义为：

$$
S_{++}^n=\{A\in R^{n\times n}:A=A^T\quad and\quad x^TAx>0\quad for\quad all\quad x\in R^n\quad such\quad that\quad x\neq 0\}
$$

</details></blockquote>

$$
p(x;\mu,\Sigma)=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)
$$

我们可以写作$x\sim\mathcal{N}(\mu,\Sigma)$。

#### 2. 高斯分布的特点

多元高斯在实践中非常方便，因为其如下的特点:

- **特点 #1：** 如果你知道以$x$为随机变量的高斯分布的均值$\mu$和协方差矩阵$\Sigma$。则你可以直接写出关于$x$的概率密度函数。

- **特点 #2：** 下列高斯积分具有闭式解(closed-form solutions)：

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

在本节中，我们将详细讨论前面描述的每个封闭属性，我们将使用特点#1和#2来证明属性，或者至少给出一些关于属性正确性的直觉。

下面是我们本节将要介绍的内容的路线图：

||独立高斯分布的和|联合高斯分布的边缘分布|联合高斯分布的条件分布|
|:-:|:-:|:-:|:-:|
|为什么是高斯函数的解释|不介绍|介绍|介绍|
|概率密度函数的结果|介绍|介绍|介绍|

##### 3.1 独立高斯分布的和是高斯分布

本规则的正式表述为：

设有$y\sim\mathcal{N}(\mu,\Sigma)$和$z\sim\mathcal{N}(\mu',\Sigma')$为独立高斯分布，其中随机变量$\mu,\mu'\in R^n$且$\Sigma,\Sigma'\in S_{++}^n$。则它们的和也同样是高斯分布：

$$
y+z\sim\mathcal{N}(\mu+\mu',\Sigma+\Sigma')
$$

在我们证明上面的结论前，先给出一些直观结果：

1. 首先要指出的是上述规则中独立假设的重要性。为了了解为什么这很重要，假设$y\sim\mathcal{N}(\mu,\sigma)$是服从于均值$\mu$方差$\sigma$的多元高斯分布，并且假设$z=-y$。很明显，$z$也是服从于与多元高斯分布（事实上，$z\sim\mathcal{N}(-\mu,\sigma)$），但是$y+z$等于零（不是高斯分布）！
2. 第二件需要指出的事情是许多学生感到困惑的一点：如果我们把两个高斯概率密度函数（多维空间中的“肿块(bumps)”）加在一起，我们会得到一些峰（即“双峰(two-humped)”的概率密度函数）么？在这里，我们要注意到随机变量$y + z$的概率密度函数并不是简单的将两个单独的概率密度函数的随机变量$y$和$z$相加，而是会变成$y$和$z$的卷积的概率密度函数。$^2$ 然而证明“两个高斯概率密度函数的卷积得到一个高斯概率密度函数”超出了这门课的范围。

<blockquote><details><summary>上一小段上标2的说明（详情请点击本行）</summary>

2 例如，如果$y$和$z$是单变量高斯函数（即：$y\sim\mathcal{N}(\mu,\sigma^2),z\sim\mathcal{N}(\mu,\sigma'^2)$），则它们的概率密度的卷积由下式给出：

$$
\begin{aligned}
p(y+z;\mu,\mu',\sigma,\sigma'^2) &=\int_{-\infin}^{\infin}p(w;\mu,\sigma^2)p(y+z-w;\mu',\sigma'^2)dw \\
&= \int_{-\infin}^{\infin}\frac 1{\sqrt{2\pi}\sigma}\exp\left(-\frac 1{2\sigma^2}(w-\mu)^2\right)\cdot \frac 1{\sqrt{2\pi}\sigma'}\exp\left(-\frac 1{2\sigma'^2}(y+z-w-\mu')^2\right)dw
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

##### 3.2 联合高斯分布的边缘分布是高斯分布

本规则的正式表述为:

假设

$$
\begin{bmatrix}x_A\\x_B\end{bmatrix}\sim\mathcal{N}\begin{pmatrix}\begin{bmatrix}\mu_A\\\mu_B\end{bmatrix},\begin{bmatrix}\Sigma_{AA}&\Sigma_{AB}\\\Sigma_{BA}&\Sigma_{BB}\end{bmatrix}\end{pmatrix}
$$

其中$x_A\in R^m,x_B\in R^n$并选择均值向量和协方差矩阵子块的维数与$x_A$和$x_B$进行匹配。则边缘概率密度函数如下所示：

$$
p(x_A)=\int_{x_B\in R^n}p(x_A,x_B;\mu,\Sigma)dx_B \\
p(x_B)=\int_{x_A\in R^m}p(x_A,x_B;\mu,\Sigma)dx_A
$$

上面式子都是高斯分布：

$$
x_A\sim\mathcal{N}(\mu_A,\Sigma_{AA}) \\
x_B\sim\mathcal{N}(\mu_B,\Sigma_{BB})
$$

为了证明这个规则，我们只关注变量$x_A$的边缘分布。

>4 一般来说，对于一个高斯分布的随机向量$x$，只要我们对均值向量的项和协方差矩阵的行/列按对应的方式进行置换，则总是可以对$x$的项进行置换。因此，只看$x_A$就足够了，$x_B$的结果也立即得到了。

首先，请注意计算边缘分布的均值和协方差矩阵很简单：只需从联合概率密度函数的均值和协方差矩阵中提取相应的子块。为了确保这是绝对清楚的，我们来看看$x_{A,i}$和$x_{A,j}$（$x_A$的第$i$个部分和$x_A$的第$j$个部分）之间的协方差。注意$x_{A,i}$和$x_{A,j}$同样也是下面式子的第$i$个和第$j$个部分：

$$
\begin{bmatrix}x_A\\x_B\end{bmatrix}
$$

（因为$x_A$出现在这个向量的上部分）。要找到它们的协方差，我们只需简单的使用下面式子的那个协方差矩阵的第$(i, j)$个元素即可：

$$
\begin{bmatrix}\Sigma_{AA}&\Sigma_{AB}\\\Sigma_{BA}&\Sigma_{BB}\end{bmatrix}
$$

第$(i, j)$个元素在可以在$\Sigma_{AA}$子块矩阵中找到。事实上就是$\Sigma_{AA,ij}$。对所有的$i,j\in \{1,\dots,m\}$使用这个参数，我们可以发现$x_A$的协方差矩阵可以简化为$\Sigma_{AA}$。类似的方法可以用来求$x_A$的均值简化为$\mu_A$。因此，上面的论证告诉我们，如果我们知道$x_A$的边缘分布是高斯分布，那么我们就可以用合适的均值子矩阵以及联合概率密度函数的协方差矩阵立即写出$x_A$的概率密度函数。

上面的论证虽然简单，但多少有些不令人满意：我们如何才能真正确定$x_A$是一个多元高斯分布？关于这一点的论述有点冗长，因此，与其节外生枝，不如先列出我们的推导过程：

1. 明确写出边缘概率密度函数的积分形式。
2. 通过对逆协方差矩阵进行分块来重写积分。
3. 使用“平方和”参数来计算$x_B$上的积分。
4. 论述得到的概率密度函数是高斯的。

下面让我们分别研究一下上面提到的每一个步骤。

###### 3.2.1 边缘概率密度函数的积分形式

假设我们想直接计算$x_A$的密度函数。然后，我们需要计算积分：

$$
\begin{aligned}
p(x_A) &= \int_{x_B\in R^n}p(x_A,x_B;\mu,\Sigma)dx_B \\
&= \frac{1}{(2\pi)^{\frac{m+n}{2}} \begin{vmatrix}\Sigma_{AA}&\Sigma_{AB}\\\Sigma_{BA}&\Sigma_{BB}\end{vmatrix}^{1/2}}\int_{x_B\in R^n}\exp\left(-\frac12\begin{bmatrix}x_A-\mu_A\\x_B-\mu_B\end{bmatrix}^T\begin{bmatrix}\Sigma_{AA}&\Sigma_{AB}\\\Sigma_{BA}&\Sigma_{BB}\end{bmatrix}^{-1}\begin{bmatrix}x_A-\mu_A\\x_B-\mu_B\end{bmatrix}\right)dx_B
\end{aligned}
$$

###### 3.2.2 逆协方差矩阵的分块

为了进一步推导，我们需要把指数中的矩阵乘积写成稍微不同的形式。特别地，让我们定义下面这个矩阵：

$$
V=\begin{bmatrix}V_{AA}&V_{AB}\\V_{BA}&V_{BB}\end{bmatrix}=\Sigma^{-1}
$$

这里我们可能会有下面这种诱人的推导想法：

$$
V=\begin{bmatrix}V_{AA}&V_{AB}\\V_{BA}&V_{BB}\end{bmatrix}=\begin{bmatrix}\Sigma_{AA}&\Sigma_{AB}\\\Sigma_{BA}&\Sigma_{BB}\end{bmatrix}^{-1}“=”\begin{bmatrix}\Sigma_{AA}^{-1}&\Sigma_{AB}^{-1}\\\Sigma_{BA}^{-1}&\Sigma_{BB}^{-1}\end{bmatrix}
$$

然而，最右边的等号并不成立！我们将在稍后的步骤中讨论这个问题；不过，现在只要将$V$定义为上述形式就足够了，而不必担心每个子矩阵的实际内容是什么。

利用$V$的这个定义，积分扩展到下面的式子：

$$
\begin{aligned}
p(x_A)=\frac 1Z\int_{x_B\in R^n}\exp(-&[\frac 12(x_A-\mu_A)^TV_{AA}(x_A-\mu_A)+\frac 12(x_A-\mu_A)^TV_{AB}(x_B-\mu_B) \\
& +\frac 12(x_B-\mu_B)^TV_{BA}(x_A-\mu_A)+\frac 12(x_B-\mu_B)^TV_{BB}(x_B-\mu_B)])dx_B
\end{aligned}
$$

其中$Z$是一个常数，不依赖于$x_A$或$x_B$，我们暂时忽略它。如果你以前没有使用过分块矩阵，那么上面的展开对你来说可能有点神奇。这类似于当定义一个二次形式基于某个矩阵$A$时，则可得：

$$
x^TAx=\sum_i\sum_jA_{ij}x_ix_j=x_1A_{11}x_1+x_1A_{12}x_2+x_2A_{21}x_1+x_2A_{22}x_2
$$

花点时间自己研究一下，上面的矩阵推广也适用。

##### 3.2.3 $x_B$上的积分

为了求积分，我们要对$x_B$积分。然而，一般来说，高斯积分是很难手工计算的。我们能做些什么来节省计算时间吗？事实上，有许多高斯积分的答案是已知的（见特点#2）。那么，本节的基本思想是将上一节中的积分转换为一种形式，在这种形式中，我们可以应用特点#2中的一个结果，以便轻松地计算所需的积分。

这其中的关键是一个数学技巧，称为“配方法(completion of squares)”。考虑二次函数  。其中

$$
\frac 12x^TAx+b^Tz+c=\frac 12(z+A^{-1}b)^TA(z+A^{-1}b)+c-\frac 12b^TA^{-1}b
$$

下面使用单变量代数中的“配方法”来泛华的多元变量的等式:

$$
\frac 12az^2+bz+c=\frac 12a(z+\frac bz)^2+c-\frac {b^2}{2a}
$$

若要将配方法应用于上述情形，令

$$
\begin{aligned}
z &= x_B-\mu_B \\
A &= V_{BB} \\
b &=V_{BA}(x_A-\mu_A) \\
c &=\frac 12(x_A-\mu_A)^TV_{AA}(x_A-\mu_A)
\end{aligned}
$$

然后，这个积分可以重写为

$$
\begin{aligned}
p(x_A)=\frac 1Z\int_{x_B\in R^n}exp(-&[\frac 12(x_B-\mu_B)^TV_{AA}(x_A-\mu_A)+\frac 12(x_A-\mu_A)^TV_{AB}(x_B-\mu_B) \\
& +\frac 12(x_B-\mu_B)^TV_{BA}(x_A-\mu_A)+\frac 12(x_B-\mu_B)^TV_{BB}(x_B-\mu_B)])dx_B
\end{aligned}
$$

我们可以提出不包括$x_B$的项，

$$
\begin{aligned}
p(x_{A})&=\exp\left(-\frac{1}{2}\left(x_{A}-\mu_{A}\right)^{T} V_{A A}\left(x_{A}-\mu_{A}\right)+\frac{1}{2}\left(x_{A}-\mu_{A}\right)^{T} V_{A B} V_{B B}^{-1} V_{B A}\left(x_{A}-\mu_{A}\right)\right) \\ 
&\quad \cdot \frac{1}{Z} \int_{x_{B} \in \mathbb{R}^{n}} \exp \left(-\frac{1}{2}\left[\left(x_{B}-\mu_{B}+V_{B B}^{-1} V_{B A}\left(x_{A}-\mu_{A}\right)\right)^{T} V_{B B}\left(x_{B}-\mu_{B}+V_{B B}^{-1} V_{B A}\left(x_{A}-\mu_{A}\right)\right)\right]\right) d x_{B}
\end{aligned}
$$

现在，我们可以应用特点#2。特别的，我们知道通常情况下随机变量为$x$多元高斯分布，如果设均值$\mu$，协方差矩阵$\Sigma$，则概率密度函数可以得到如下式子：

$$
\frac{1}{(2 \pi)^{n / 2}|\Sigma|^{1 / 2}} \int_{\mathbf{R}^{n}} \exp \left(-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right)=1
$$

或等价与下式：

$$
\int_{R^{n}} \exp \left(-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right)=(2 \pi)^{n / 2}|\Sigma|^{1 / 2}
$$

我们用这个事实来消去表达式中剩下的积分以得到$p(x_A)$：

$$
p\left(x_{A}\right)=\frac{1}{Z} \cdot(2 \pi)^{n / 2}\left|V_{B B}\right|^{1 / 2} \cdot \exp \left(-\frac{1}{2}\left(x_{A}-\mu_{A}\right)^{T}\left(V_{A A}-V_{A B} V_{B B}^{-1} V_{B A}\right)\left(x_{A}-\mu_{A}\right)\right)
$$

###### 3.2.4 论述得到的概率密度函数是高斯函数

这时我们几乎已经完成了全部计算！忽略前面的归一化常数，我们看到$x_A$的概率密度函数是$x_A$的二次形的指数。我们可以很快意识到概率密度函数就是均值向量为$\mu_A$，协方差矩阵为$(V_{A A}-V_{A B} V_{B B}^{-1} V_{B A})^{-1}$的高数分布。虽然协方差矩阵的形式看起来有点复杂，但是我们已经完成了我们开始想要展示的概念——即$x_A$有一个边缘高斯分布。利用前面的逻辑，我们可以得出这个协方差矩阵必须以某种方式消去$\Sigma_{AA}$。

但是，如果你好奇，也可以证明我们的推导与之前的证明是一致的。为此，我们对分块矩阵使用以下结果：

$$
\left[ \begin{array}{cc}{A} & {B} \\ {C} & {D}\end{array}\right]^{-1}=\left[ \begin{array}{cc}{M^{-1}} & {-M^{-1} B D^{-1}} \\ {-D^{-1} C M^{-1}} & {D^{-1}+D^{-1} C M^{-1} B D^{-1}}\end{array}\right]
$$

其中$M=A-B D^{-1} C$。这个公式可以看作是$2\times 2$矩阵显式逆矩阵的多变量推广：

$$
\left[ \begin{array}{ll}{a} & {b} \\ {c} & {d}\end{array}\right]^{-1}=\frac{1}{a d-b c} \left[ \begin{array}{cc}{d} & {-b} \\ {-c} & {a}\end{array}\right]
$$

用这个公式，可以得出：

$$
\begin{aligned}
\left[ \begin{array}{cc}{\Sigma_{A A}} & {\Sigma_{A B}} \\ {\Sigma_{B A}} & {\Sigma_{B B}}\end{array}\right] &=\left[ \begin{array}{ll}{V_{A A}} & {V_{A B}} \\ {V_{B A}} & {V_{B B}}\end{array}\right]^{-1} \\
&=\left[ \begin{array}{cc}{\left(V_{A A}-V_{A B} V_{B B}^{-1} V_{B A}\right)^{-1}} & {-\left(V_{A A}-V_{A B} V_{B B}^{-1} V_{B A}\right)^{-1} V_{A B} V_{B B}^{-1}} \\ {-V_{B B}^{-1} V_{B A}\left(V_{A A}-V_{A B} V_{B B}^{-1} V_{B A}\right)^{-1}} & {\left(V_{B B}-V_{B A} V_{A A}^{-1} V_{A B}\right)^{-1}}\end{array}\right]
\end{aligned}
$$

正如我们所期望的那样，我们马上就能得出$\left(V_{A A}-V_{A B} V_{B B}^{-1} V_{B A}\right)^{-1}=\Sigma_{A A}$。

#### 3.3 联合高斯分布的条件分布是高斯分布

本规则的正式表述为:

假设：

$$
\left[ \begin{array}{l}{x_{A}} \\ {x_{B}}\end{array}\right]\sim\mathcal{N}\left(\left[ \begin{array}{l}{\mu_{A}} \\ {\mu_{B}}\end{array}\right], \left[ \begin{array}{cc}{\Sigma_{A A}} & {\Sigma_{A B}} \\ {\Sigma_{B A}} & {\Sigma_{B B}}\end{array}\right]\right)
$$

其中$x_{A} \in \mathbf{R}^{m}, x_{B} \in \mathbf{R}^{n}$，并选择均值向量和协方差矩阵子块的维数来匹配$x_A$和$x_B$。则条件概率密度函数为：

$$
\begin{aligned} p\left(x_{A} | x_{B}\right) &=\frac{p\left(x_{A}, x_{B} ; \mu, \Sigma\right)}{\int_{x_{A} \in \mathbb{R}^{m}} p\left(x_{A}, x_{B} ; \mu, \Sigma\right) d x_{A}} \\ p\left(x_{B} | x_{A}\right) &=\frac{p\left(x_{A}, x_{B} ; \mu, \Sigma\right)}{\int_{x_{B} \in \mathbb{R}^{n}} p\left(x_{A}, x_{B} ; \mu, \Sigma\right) d x_{B}} \end{aligned}
$$

同样是高斯分布：

$$
\begin{array}{l}{x_{A}\left|x_{B} \sim \mathcal{N}\left(\mu_{A}+\Sigma_{A B} \Sigma_{B B}^{-1}\left(x_{B}-\mu_{B}\right), \Sigma_{A A}-\Sigma_{A B} \Sigma_{B B}^{-1} \Sigma_{B A}\right)\right.} \\ {x_{B} | x_{A} \sim \mathcal{N}\left(\mu_{B}+\Sigma_{B A} \Sigma_{A A}^{-1}\left(x_{A}-\mu_{A}\right), \Sigma_{B B}-\Sigma_{B A} \Sigma_{A A}^{-1} \Sigma_{A B}\right)}\end{array}
$$

和之前一样，我们只研究条件分布$x_B|x_A$，另一个结果是对称的。我们的推导过程如下:

1. 明确写出条件概率密度函数的表达式。
2. 通过划分逆协方差矩阵重写表达式。
3. 使用“平方和”参数。
4. 论述得到的概率密度函数是高斯函数。

下面让我们分别研究一下上面提到的每一个步骤。

###### 3.3.1 明确写出条件概率密度函数的表达式

假设我们想直接计算给定$x_A$下$x_B$的概率密度函数。则我们需要计算下式：

$$
\begin{aligned}
p\left(x_{B} | x_{A}\right) &=\frac{p\left(x_{A}, x_{B} ; \mu, \Sigma\right)}{\int_{x_{B} \in R^m} p\left(x_{A}, x_{B} ; \mu, \Sigma\right) d x_{A}} \\
&=\frac{1}{Z^{\prime}} \exp \left(-\frac{1}{2} \left[ \begin{array}{c}{x_{A}-\mu_{A}} \\ {x_{B}-\mu_{B}}\end{array}\right]^{T} \left[ \begin{array}{cc}{\Sigma_{A A}} & {\Sigma_{A B}} \\ {\Sigma_{B A}} & {\Sigma_{B B}}\end{array}\right]^{-1} \left[ \begin{array}{c}{x_{A}-\mu_{A}} \\ {x_{B}-\mu_{B}}\end{array}\right]\right)
\end{aligned}
$$

其中$Z'$是一个归一化常数，我们用该常数表达不依赖于$x_B$的因子。注意，这一次，我们甚至不需要计算任何积分——积分的值不依赖于$x_B$，因此积分可以化简成归一化常数$Z'$。

###### 3.3.2 通过划分逆协方差矩阵重写表达式

和之前一样，我们用矩阵$V$重新参数化概率密度函数，由此得到下式：

$$
\begin{aligned}
p\left(x_{B} | x_{A}\right) &=\frac{1}{Z^{\prime}} \exp \left(-\frac{1}{2} \left[ \begin{array}{c}{x_{A}-\mu_{A}} \\ {x_{B}-\mu_{B}}\end{array}\right]^{T} \left[ \begin{array}{cc}{V_{A A}} & {V_{A B}} \\ {V_{B A}} & {V_{B B}}\end{array}\right] \left[ \begin{array}{c}{x_{A}-\mu_{A}} \\ {x_{B}-\mu_{B}}\end{array}\right]\right) \\
&=\frac{1}{Z^{\prime}} \exp (-[\frac{1}{2}\left(x_{A}-\mu_{A}\right)^{T} V_{A A}\left(x_{A}-\mu_{A}\right)+\frac{1}{2}\left(x_{A}-\mu_{A}\right)^{T} V_{A B}\left(x_{B}-\mu_{B}\right) \\
&\qquad\qquad\qquad+\frac{1}{2}\left(x_{B}-\mu_{B}\right)^{T} V_{B A}\left(x_{A}-\mu_{A}\right)+\frac{1}{2}\left(x_{B}-\mu_{B}\right)^{T} V_{B B}\left(x_{B}-\mu_{B}\right) ] )
\end{aligned}
$$

###### 3.3.3 使用“平方和”参数

回忆下面这个式子：

$$
\frac{1}{2} z^{T} A z+b^{T} z+c=\frac{1}{2}\left(z+A^{-1} b\right)^{T} A\left(z+A^{-1} b\right)+c-\frac{1}{2} b^{T} A^{-1} b
$$

假设$A$是一个对称的非奇异矩阵。如前所述，要将平方的补全应用于上述情况，令：

$$
\begin{aligned} 
z &=x_{B}-\mu_{B} \\ 
A &=V_{B B} \\ 
b &=V_{B A}\left(x_{A}-\mu_{A}\right) \\ 
c &=\frac{1}{2}\left(x_{A}-\mu_{A}\right)^{T} V_{A A}\left(x_{A}-\mu_{A}\right) 
\end{aligned}
$$

然后，可以将$p(x_B | x_A)$的表达式重写为：

$$
\begin{array}{c}{p\left(x_{B} | x_{A}\right)=\frac{1}{Z^{\prime}} \exp \left(-\left[\frac{1}{2}\left(x_{B}-\mu_{B}+V_{B B}^{-1} V_{B A}\left(x_{A}-\mu_{A}\right)\right)^{T} V_{B B}\left(x_{B}-\mu_{B}+V_{B B}^{-1} V_{B A}\left(x_{A}-\mu_{A}\right)\right)\right.\right.} \\ 
{+\frac{1}{2}\left(x_{A}-\mu_{A}\right)^{T} V_{A A}\left(x_{A}-\mu_{A}\right)-\frac{1}{2}\left(x_{A}-\mu_{A}\right)^{T} V_{A B} V_{B B}^{-1} V_{B A}\left(x_{A}-\mu_{A}\right) ] )}\end{array}
$$

将不依赖于$x_B$的指数部分化简到归一化常数中，得到：

$$
p\left(x_{B} | x_{A}\right)=\frac{1}{Z^{\prime \prime}} \exp \left(-\frac{1}{2}\left(x_{B}-\mu_{B}+V_{B B}^{-1} V_{B A}\left(x_{A}-\mu_{A}\right)\right)^{T} V_{B B}\left(x_{B}-\mu_{B}+V_{B B}^{-1} V_{B A}\left(x_{A}-\mu_{A}\right)\right)\right)
$$

###### 3.3.4 论述得到的概率密度函数是高斯函数

看最后一个表达式，表达式$p(x_B|x_A)$是均值为$\mu_B-V_{B B}^{-1} V_{B A}\left(x_{A}-\mu_{A}\right)$，协方差矩阵为$V_{B B}^{-1}$的高斯概率密度函数。像往常一样，回忆一下矩阵等式：

$$
\left[ \begin{array}{cc}{\Sigma_{A A}} & {\Sigma_{A B}} \\ {\Sigma_{B A}} & {\Sigma_{B B}}\end{array}\right]=
\left[ \begin{array}{c}{\left(V_{A A}-V_{A B} V_{B B}^{-1} V_{B A}\right)^{-1}}&-\left(V_{A A}-V_{A B} V_{B B}^{-1} V_{B A}\right)^{-1} V_{A B} V_{B B}^{-1} \\ {-V_{B B}^{-1} V_{B A}\left(V_{A A}-V_{A B} V_{B B}^{-1} V_{B A}\right)^{-1}}&\left(V_{B B}-V_{B A} V_{A A}^{-1} V_{A B}\right)^{-1}\end{array}\right]
$$

从上式可以推出：

$$
\mu_{B | A}=\mu_{B}-V_{B B}^{-1} V_{B A}\left(x_{A}-\mu_{A}\right)=\mu_{B}+\Sigma_{B A} \Sigma_{A A}^{-1}\left(x_{A}-\mu_{A}\right)
$$

反过来，我们也可以利用矩阵恒等式得到：

$$
\left[ \begin{array}{cc}{V_{A A}} & {V_{A B}} \\ {V_{B A}} & {V_{B B}}\end{array}\right]=
\left[ \begin{array}{c}{\left(\Sigma_{A A}-\Sigma_{A B} \Sigma_{B B}^{-1} \Sigma_{B A}\right)^{-1}}&-\left(\Sigma_{A A}-\Sigma_{A B} \Sigma_{A A}^{-1} \Sigma_{B B}\right)^{-1} \Sigma_{A B} \Sigma_{B B}^{-1} \\ {-\Sigma_{B B}^{-1} \Sigma_{B A}\left(\Sigma_{A A}-\Sigma_{A B} \Sigma_{B B}^{-1} \Sigma_{B A}\right)^{-1}}&\left(\Sigma_{B B}-\Sigma_{B A} \Sigma_{A A}^{-1} \sum_{A B}\right)^{-1}\end{array} \right]
$$

由此推出：

$$
\Sigma_{B | A}=V_{B B}^{-1}=\Sigma_{B B}-\Sigma_{B A} \Sigma_{A A}^{-1} \Sigma_{A B}
$$

我们完成了!

#### 4. 总结

在本节的笔记中，我们使用了多元高斯的一些简单性质（加上一些矩阵代数技巧）来证明多元高斯分布满足许多封闭性质。一般来说，多元高斯分布是概率分布非常有用的表示形式，因为封闭性保证了这一点：即我们所希望的那样使用多元高斯分布执行的大多数类型的操作都可以以封闭形式完成。从分析的角度来看，涉及多元高斯的积分在实际应用中是往往是很好计算的，因为我们可以依赖于已知的高斯积分来避免自己进行积分。

#### 5. 练习

理解题：令$A\in R^{n\times n}$是对称非奇异方阵，$b\in R^n,c$，证明：

$$
\int_{x \in \mathbf{R}^{n}} \exp \left(-\frac{1}{2} x^{T} A x-x^{T} b-c\right) d x=\frac{(2 \pi)^{n / 2}}{|A|^{1 / 2} \exp \left(c-b^{T} A^{-1} b\right)}
$$

##### 参考资料

>有关多元高斯的更多信息，请参见:
Bishop, Christopher M. Pattern Recognition and Machine Learning. Springer,2006.