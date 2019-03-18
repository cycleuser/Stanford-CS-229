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


### 凸优化概述

在上一篇的笔记中，我们开始学习**凸优化，** 并且学习到了如下优化问题的数学形式：

$$
\begin{aligned}
\min_{x\in R^n}\quad&f(x) \\
subject\quad to\quad &x\in C \qquad\qquad (1)
\end{aligned}
$$

在凸优化问题的设定中，$x\in R^n$是一个被称作**优化变量**的向量。$f:R^n\rightarrow R$是我们想要优化的**凸函数。** $C\subseteq R^n$是一个被称作可行域集合的**凸集。** 从计算的角度来看，凸优化问题的有趣之处在于，任何局部最优解总是保证其也是全局最优的解。在过去的几十年里，求解凸优化问题的通用方法也变得越来越可靠有效。

在后面的课堂笔记中，我们继续探索凸优化领域。特别地，我们将探讨凸优化理论中一个强大的概念，称为拉格朗日对偶。重点研究拉格朗日对偶的主要思想和机制；特别地，我们将描述拉格朗日的概念以及它与原问题和对偶问题之间的关系，我们还会介绍Karush-Kuhn-Tucker (KKT)条件在提供凸优化问题最优解的充要条件中的作用。

#### 1. 拉格朗日对偶

一般来说，拉格朗日对偶理论是研究凸优化问题的最优解。正如我们在之前的课上看到的，当最小化一个关于$x\in R^n$的可微凸函数$f(x)$时，使得解集$x^*\in R^n$是全局最优解的一个充要条件是$\nabla_xf(x^*)=0$。然而，在具有约束条件的凸优化问题的一般设置中，这种简单的最优性条件并不适用。对偶理论的一个主要目标是用严格的数学方法描述凸规划的最优点。

在后面的笔记中，我们简要介绍了拉格朗日对偶性及其在给出了如下一般可微凸优化问题形式时的应用：

$$
\begin{aligned}
\min_{x\in R^n}\quad&f(x) \\
subject\quad to\quad &g_i(x)\le 0,\quad i=1,\dots,m, \qquad\qquad (OPT) \\
&h_i(x)=0,\quad i=1,\dots,p,
\end{aligned}
$$

其中$x\in R^n$是**优化变量**，$f:R^n\rightarrow R$以及$g_i:R^n\rightarrow R$是**可微凸函数$^1$。**$h_i:R^n\rightarrow R$是**仿射函数。$^2$**

>1 回忆一下我们称一个函数$f:S\rightarrow R$是一个凸函数，需要满足给定任意$x,y\in S$以及$0\le\theta\le 1$，都有$f(\theta x +(1-\theta) y)\le \theta f(x)+(1-\theta)f(y)$成立。如果函数$-f$是凸函数，则函数$f$是凹函数。

>2 回忆一下仿射函数有如下的形式$f(x)=b^Tx+c$，满足$b\in R^n,c\in R$。由于仿射函数的海森矩阵是一个零矩阵（即该矩阵是半正定也是半负定矩阵），因此仿射函数即是凸函数，也是凹函数

##### 1.1 拉格朗日函数

在这一节中，我们介绍了拉格朗日对偶理论的基础——拉格朗日函数。给出了凸约束极小化问题的形式，（广义）拉格朗日是一个函数$\mathcal{L}:R^n\times R^m\times R^p\rightarrow R$，定义如下：

$$
\mathcal{L}(x,\alpha,\beta)=f(x)+\sum_{i=1}^m\alpha_ig_i(x)+\sum_{i=1}^p\beta_ih_i(x)\qquad\qquad (2)
$$

这里，拉格朗日函数的第一个参数是一个向量$x\in R^n$，其维数与原优化问题中的优化变量的维数相匹配；根据习惯，我们称$x$为拉格朗日函数的**原始变量(primal variables)。** 拉格朗日函数的第二个参数是一个向量$\alpha\in R^m$，对应于原优化问题中的$m$个凸不等式约束，每个约束都有一个变量$\alpha_i$。拉格朗日函数的第三个参数是是个向量$\beta\in R^p$，对应于原始优化问题的$p$个仿射不等式，每一个约束都有一个变量$\beta_i$。这些$\alpha,\beta$元素合起来被称作拉格朗日函数或**拉格朗日乘数(Lagrange multipliers)** 的**对偶变量(dual variables)**。 

直观地，拉格朗日函数可以看作是原始凸优化问题的目标函数的一个修正版本，该优化问题考虑了每个约束条件。作为拉格朗日乘数的$\alpha_i,\beta_i$可以认为是违反不同的限制条件的“代价”。拉格朗日对偶理论背后的关键直觉如下：

对于任何凸优化问题，总是存在对偶变量的设置，使得拉格朗日关于原变量的无约束极小值（保持对偶变量不变）与原约束极小化问题的解一致。

我们在第1.6节描述KKT条件时将这种直觉形式化。

##### 1.2 原问题与对偶问题

为了说明拉格朗日问题与原凸优化问题之间的关系，我们引入了与拉格朗日问题相关的原问题和对偶问题的概念：

<u>原问题</u>

考虑如下的优化问题：

$$
\min_x\underbrace{[\max_{\alpha,\beta:\alpha_i\ge0,\forall i}\mathcal{L}(x,\alpha,\beta)]}_{这部分称作\theta_\mathcal{P}(x)}=\min_x\theta_\mathcal{P}(x)\qquad\qquad(P)
$$

上面的等式中，函数$\theta_\mathcal{P}:R^n\rightarrow R$被称作**原目标(primal objective,)，** 右边的无约束极小化问题称为**原问题(primal problem)。** 在通常情况下，当$g_i(x)\le 0,i=1,\dots,m$以及$h_i(x)=0,i=1,\dots,p$时，一个点$x\in R^n$被称作**原可行域(primal feasible)。** 我们通常使用向量$x^*\in R^n$代表$(P)$式的解。我们令$p^*=\theta_\mathcal{P}(x^*)$代表原目标的最优值。

<u>对偶问题</u>

通过对上述最小化和最大化顺序的转换，我们得到了一个完全不同的优化问题：

$$
\max_{\alpha,\beta:\alpha_i\ge0,\forall i}\underbrace{[\min_x\mathcal{L}(x,\alpha,\beta)]}_{这部分称作\theta_\mathcal{D}(\alpha,\beta)}=\max_{\alpha,\beta:\alpha_i\ge 0,\forall i}\theta_\mathcal{D}(\alpha,\beta)\qquad\qquad(D)
$$

这里的函数$\theta_\mathcal{D}:R^m\times R^p\rightarrow R$被称作**对偶目标(dual objective)，** 右边的约束最大化问题称为**对偶问题(dual problem)。** 通常情况下，当$\alpha_i\ge 0,i=1,\dots,m$时，我们称$(\alpha,\beta)$为**对偶可行域(dual feasible)。** 我们通常使用向量对$(\alpha^*,\beta^*)\in R^m\times R^p$代表$(D)$式的解。我们令$\theta_\mathcal{D}(\alpha^*,\beta^*)$代表对偶目标的最优值。

##### 1.3 原问题的解释

首先观察到，原目标函数$\theta_\mathcal{P}(x)$是一个关于$x$的凸函数$^3$。为了解释原问题，注意到：

<blockquote><details><summary>3 原目标函数是凸函数的原因（详情请点击本行）</summary>
为了解释原因，注意到：

$$
\theta_\mathcal{P}(x) = \max_{\alpha,\beta:\alpha_i\ge0,\forall i}\mathcal{L}(x,\alpha,\beta) = \max_{\alpha,\beta:\alpha_i\ge0,\forall i}[f(x)+\sum_{i=1}^m\alpha_ig_i(x)+\sum_{i=1}^p\beta_ih_i(x)]\qquad\qquad(3)
$$

从这个式子中，我们可以观察到$g_i(x)$是一个关于$x$的凸函数。因为$\alpha_i$被限制为非负数，所以对于所有的$i$，都满足$\alpha_ig_i(x)$是凸函数。类似的，因为$h_i(x)$是线性函数，所以每一个$\beta_ih_i(x)$都是关于$x$（不用管$\beta_i$的符号）的凸函数。由于凸函数的和总是凸函数，我们可以得出括号内的整体是一个关于$x$的凸函数。最后，凸函数集合的最大值也是一个凸函数（自己证明一下！），因此我们可以得出$\theta_\mathcal{P}(x)$是一个关于$x$的凸函数。
</details></blockquote>

$$
\begin{aligned}
\theta_\mathcal{P}(x) &= \max_{\alpha,\beta:\alpha_i\ge0,\forall i}\mathcal{L}(x,\alpha,\beta) \qquad\qquad&(4)\\
&= \max_{\alpha,\beta:\alpha_i\ge0,\forall i}[f(x)+\sum_{i=1}^m\alpha_ig_i(x)+\sum_{i=1}^p\beta_ih_i(x)]&(5)\\
&=f(x)+\max_{\alpha,\beta:\alpha_i\ge0,\forall i}[\sum_{i=1}^m\alpha_ig_i(x)+\sum_{i=1}^p\beta_ih_i(x)]&(6)
\end{aligned}
$$

可以看到这样一个事实：函数$f(x)$不依赖于$\alpha$或者$\beta$。只考虑括号内的符号，可以注意到：

- 如果任意$g_i(x)>0$，则使括号内表达式最大化需要使对应的$\alpha_i$为任意大的正数；但是，如果$g_i(x)\le 0$，且需要$\alpha_i$非负，这就意味着调节$\alpha_i$达到整体最大值的设置为$\alpha_i= 0$，此时的最大值为$0$。

- 类似地，如果任意$h_i(x) \ne 0$，则要使括号内表达式最大化，需要选择与$h_i(x)$符号相同且任意大的对应$\beta_i$；但是，如果$h_i(x)=0$，则最大值与$\beta_i$无关，只能取$0$。

把这两种情况放在一起，我们看到如果$x$是在原可行域内（即$g_i(x)\le 0,i=1,\dots,m$以及$h_i(x)=0,i=1,\dots,p$）的，则括号内表达式的最大值为$0$，但如果违反任何约束，则最大值为$\infin$。根据前面的讨论，我们可以写出以下的式子：

$$
\theta_\mathcal{P}(x) = \underbrace{f(x)}_{原目标(original\quad objective)} + \underbrace{\begin{cases}
0& 如果x在原始可行域内\\
\infin& 如果x不在原始可行域内
\end{cases}
}_{为了“跨域(carving\quad away)”不可行解的障碍函数}\qquad(7)
$$

因此，我们可以将原始目标$\theta_\mathcal{P}(x)$理解为原问题凸目标函数的一个修正版本，其区别是不可行解（即那些违法限制条件的$x$的集合）含有目标值$\infin$。直观地说，我们可以考虑如下式子：

$$
\max_{\alpha,\beta:\alpha_i\ge0,\forall i}[\sum_{i=1}^m\alpha_ig_i(x)+\sum_{i=1}^p\beta_ih_i(x)]=\begin{cases}
0& 如果x在原始问题的可行域内\\
\infin& 如果x不在原始问题可行域内
\end{cases}\qquad(8)
$$

作为一种障碍函数(“barrier” function)，它防止我们将不可行点作为优化问题的候选解。

##### 1.4 对偶问题的解释

对偶目标函数$\theta_\mathcal{D}(\alpha,\beta)$是一个关于$\alpha$和$\beta$的凸函数$^4$。为了解释对偶问题，我们首先做如下观察:

<blockquote><details><summary>4 对偶目标函数是凸函数的原因（详情请点击本行）</summary>
为了解释原因，注意到：

$$
\theta_\mathcal{D}(\alpha,\beta) = \min_{x}\mathcal{L}(x,\alpha,\beta) = \min_{x}[f(x)+\sum_{i=1}^m\alpha_ig_i(x)+\sum_{i=1}^p\beta_ih_i(x)]\qquad\qquad(9)
$$

从这个式子中，我们可以观察到对于任意固定值$x$，等式中括号里面是一个关于$\alpha$和$\beta$的仿射函数，因此是凸函数。由于凹函数集合的最小值也是凹函数，我们可以得出结论$\theta_\mathcal{D}(\alpha,\beta)$是一个关于$\alpha$和$\beta$的凸函数
</details></blockquote>

**引理 1** 如果$(\alpha,\beta)$为对偶可行域(dual feasible)。则$\theta_\mathcal{D}(\alpha,\beta)\le p^*$。

<i>证明。</i>观察如下的式子：

$$
\begin{aligned}
\theta_\mathcal{D}(\alpha,\beta) &= \min_{x}\mathcal{L}(x,\alpha,\beta) \qquad\qquad&(10)\\
&\le \mathcal{L}(x^*,\alpha,\beta)&(11)\\
&= f(x^*)+\sum_{i=1}^m\alpha_ig_i(x^*)+\sum_{i=1}^p\beta_ih_i(x^*)&(12)\\
&\le f(x^*)=p^*&(13)
\end{aligned}
$$

这里，第一步和第三步分别直接遵循对偶目标函数和拉格朗日函数的定义。第二步是根据不能等号前面的表达式的意思是$x$在的所有可能值上使得函数$\mathcal{L}(x,\alpha,\beta)$为最小的那个值，最后一步是根据$x^*$是在原可行域内的这个事实得出的，并且等式$(8)$也暗示等式$(12)$的后两项必须是非正数。

引理表明，给定任何对偶可行的$(\alpha,\beta)$，对偶目标$\theta_\mathcal{D}(\alpha,\beta)$提供了原问题优化值$p^*$的一个下界。由于对偶问题涉及到在所有对偶可行域$(\alpha,\beta)$上使对偶目标最大化。因此，对偶问题可以看作是对可能的最紧下界$p^*$的搜索。这就对任何原始和对偶优化问题对产生了一个性质，这个性质被称为**弱对偶(weak duality)：**

**引理 2** （弱对偶）。对于任何一对原始的和对偶的问题，$d^*\le p^*$。

显然，弱对偶性是引理$1$使用$(\alpha,\beta)$作为对偶可行点的结果。对于一些原始/对偶优化问题，一个更强的可以成立的结果，其被称为**强对偶性(strong duality)：**

**引理 3** （强对偶）。对于满足一定被称为**约束规定(constraint qualifications)** 的技术条件的任何一对原问题和对偶问题， 可得到$d^*= p^*$。

存在许多不同的约束条件，其中最常调用的约束条件称为**Slater条件(Slater’s condition)：** 如果存在一个所有不等式约束（即$g_i(x)<0,i=1,\dots,m$）都严格满足的可行原始解$x$，则原始/对偶问题对满足Slater条件。在实际应用中，几乎所有的凸问题都满足某种约束条件，因此原问题和对偶问题具有相同的最优值。

##### 1.5 互补松弛性 

凸优化问题强对偶性的一个特别有趣的结果是**互补松弛性(complementary slackness)**（或KKT互补）：

**引理 4** （互补松弛性）。如果强对偶成立，则对于每一个$i=1,\dots,m$都有$\alpha_i^*g(x_i^*)=0$

<i>证明。</i>假设强对偶性成立。主要是复制上一节的证明，注意这下面的式子：

$$
\begin{aligned}
p^*=d^*=\theta_\mathcal{D}(\alpha^*,\beta^*) &= \min_{x}\mathcal{L}(x,\alpha^*,\beta^*) \qquad\qquad&(14)\\
&\le \mathcal{L}(x^*,\alpha^*,\beta^*)&(15)\\
&= f(x^*)+\sum_{i=1}^m\alpha_i^*g_i(x^*)+\sum_{i=1}^p\beta_i^*h_i(x^*)&(16)\\
&\le f(x^*)=p^*&(17)
\end{aligned}
$$

由于这个一系列式子中的第一个和最后一个表达式是相等的，因此每个中间表达式也是相等的。从式子$(16)$减去式子$(17)$的左半边，我们得到：

$$
\sum_{i=1}^m\alpha_i^*g_i(x^*)+\sum_{i=1}^p\beta_i^*h_i(x^*)=0\qquad\qquad(18)
$$

但是，回忆一下由于$x^*$和$(\alpha^*,\beta^*)$分别都在原可行域和对偶可行域内，所以每个$\alpha_i^*$是非负的，每个$g_i(x^*)$是非负的，以及每个$h_i(x^*)$都是零。因此，$(18)$表示的是所有非正项之和等于零的一个式子。很容易得出结论，求和中的所有单独项本身都必须为零（因为如果不为零，求和中就没有允许总体和保持为零的补偿正项）。

互补松弛性可以用许多等价的方式来表示。一种特别的方法是如下的条件对：

$$
\begin{aligned}
\alpha_i^*>0 &\Longrightarrow g_i(x^*)=0\qquad\qquad &(19) \\
g_i(x^*)<0 &\Longrightarrow\alpha_i^*=0\qquad\qquad &(20)
\end{aligned}
$$

在这个形式中，我们可以看到，无论何时任意$\alpha_i^*$都严格大于零，因此这就意味着相应的不等式约束必须保证等式成立。我们将其称为**有效约束(active constraint)。** 在支持向量机(SVMs)的情况下，有效约束也称为**支持向量(support vectors)。**

##### 1.6 KKT条件

最后，根据到目前为止的所有条件，我们就可以描述原始对偶优化对的最优条件。我们有如下的定理：

**定理 1.1** 假设$x^*\in R^n,\alpha^*\in R^m,\beta^*\in R^p$满足以下条件：

1. （原始的可行性）$g_i(x^*)\le 0,i=1,\dots,m$以及$h_i(x^*)=0,i=1,\dots,p$，
2. （对偶可行性）$\alpha_i^*\ge 0,i= 1,\dots,m$，
3. （互补松弛性）$\alpha_i^*g_i(x^*)=0,i= 1,\dots,m$，
4. （拉格朗日稳定性）$\nabla_x\mathcal{L}(x^*,\alpha^*,\beta^*)=0$。

$x^*$是原优化，$(\alpha^*,\beta^*)$是对偶优化。更进一步，如果强对偶成立，则任意原优化$x^*$以及对偶优化$(\alpha^*,\beta^*)$必须满足条件$1$到条件$4$。

这些条件被称为Karush-Kuhn-Tucker (KKT)条件。$^5$

>5 顺便提一下，KKT定理有一段有趣的历史。这个结果最初是由卡鲁什在1939年的硕士论文中推导出来的，但直到1950年被两位数学家库恩和塔克重新发现，才引起人们的注意。约翰在1948年也推导出了本质上相同结果的一个变体。关于为什么这个结果在近十年中有如此多的迭代版本都被忽视的有趣历史解释，请看这篇论文：
Kjeldsen, T.H. (2000) A contextualized historical analysis of the Kuhn-Tucker Theorem in nonlinear programming: the impact of World War II. Historica Mathematics 27: 331-361.

#### 2 一个简单的对偶实例

作为对偶的一个简单应用，在本节中，我们将展示如何形成一个简单凸优化问题的对偶问题。考虑如下的凸优化问题：

$$
\begin{aligned}
\min_{x\in R^2}\quad &x_1^2+x_2 \\
subject\quad to \quad&2x_1+x_2\ge 4 \\
& x_2\ge 1
\end{aligned}
$$

首先，我们将优化问题重写为标准形式：

$$
\begin{aligned}
\min_{x\in R^2}\quad &x_1^2+x_2 \\
subject\quad to \quad&4-2x_1-x_2\le 0 \\
& 1-x_2\le 0
\end{aligned}
$$

拉格朗日函数是：

$$
\mathcal{L}(x,\alpha)=x_1^2+x_2+\alpha_1(4-2x_1-x_2)+\alpha_2(1-x_2),\qquad\qquad (21)
$$

对偶问题的目标定义为：

$$
\theta_\mathcal{D}(\alpha)=\min_x\mathcal{L}(x,\alpha)
$$

为了用只依赖于$\alpha$（而不是$x$）的形式来表示对偶目标，我们首先观察到拉格朗日函数关于$x$是可微的，事实上，$x_1$和$x_2$（即我们可以分别求出它们的最小值）是可以分离的。

为了使函数关于$x_1$最小化，可以观察到拉格朗日函数是关于$x_1$的严格凸二次函数，因此通过将导数设为零可以找到关于$x_1$的最小值：

$$
\frac{\partial}{\partial x_1}\mathcal{L}(x,\alpha)=2x_1-2\alpha_1=0\Longrightarrow x_1=\alpha_1\qquad\qquad (22)
$$

为了使函数关于$x_2$最小化，可以观察到拉格朗日函数是$x_2$的仿射函数，其中线性系数恰好是拉格朗日系数关于$x_2$的导数：

$$
\frac{\partial}{\partial x_2}\mathcal{L}(x,\alpha)=1-\alpha_1-\alpha_2\qquad\qquad (23)
$$

如果线性系数非零，则目标函数可以通过选择与线性系数符号相反的$x_2$和任意大的增幅使其任意小。然而，如果线性系数为零，则目标函数不依赖于$x_2$。

把以上这些观察结果放在一起，我们得到：

$$
\begin{aligned}
\theta_\mathcal{D}(\alpha)&=\min_x\mathcal{L}(x,\alpha) \\
&=\min_{x_2}[\alpha_1^2+x_2+\alpha_1(4-2x_1-x_2)+\alpha_2(1-x_2)] \\
&=\min_{x_2}[-\alpha_1^2+4\alpha_1+\alpha_2+x_2(1-\alpha_1-\alpha_2)] \\
&=\begin{cases}
-\alpha_1^2+4\alpha_1+\alpha_2 \quad &如果1-\alpha_1-\alpha_2=0\\
-\infin &其他情况
\end{cases}
\end{aligned}
$$

所以对偶问题由下式给出:

$$
\begin{aligned}
\max_{x\in R^2}\quad &\theta_\mathcal{D}(\alpha) \\
subject\quad to \quad&\alpha_1\ge 0 \\
& \alpha_2\ge 0
\end{aligned}
$$

最后，我们可以通过观察使对偶约束显式$^6$的化简对偶问题：

>6 这就是说，我们把使$\theta_\mathcal{D}(\alpha)$为$-\infin$的条件移到对偶优化问题的约束集中。

$$
\begin{aligned}
\max_{x\in R^2}\quad &-\alpha_1^2+4\alpha_1+\alpha_2 \\
subject\quad to \quad&\alpha_1\ge 0 \\
& \alpha_2\ge 0 \\
& 1-\alpha_1-\alpha_2=0
\end{aligned}
$$

注意对偶问题是以为$\alpha$变量的一个凹二次规划问题。

#### 3 SVM$L_1$范数的软边界

为了看到一个更复杂的拉格朗日对偶例子，我们来推导以前课堂上给出的SVM$L_1$范数的软边界的原对偶问题，以及相应的KKT互补（即，互补松弛）条件。我们有：

$$
\begin{aligned}
\min_{w,b,\xi} \quad & \frac 12 \parallel w\parallel^2+C\sum^m_{i=1}\xi_i \\
subject\quad to \quad& y^{(i)}(w^Tx^{(i)}+b) \geq1-\xi_i,\quad &i=1,...,m\\
& \xi_i \geq 0, &i=1,...,m
\end{aligned}
$$

首先，我们使用“$\le 0$”的不等式形式把它化成标准形式：

$$
\begin{aligned}
\min_{w,b,\xi} \quad & \frac 12 \parallel w\parallel^2+C\sum^m_{i=1}\xi_i \\
subject\quad to \quad& 1-\xi_i-y^{(i)}(w^Tx^{(i)}+b) \le 0,\quad &i=1,...,m\\
& -\xi_i \le 0, &i=1,...,m
\end{aligned}
$$

接下来，我们构造广义拉格朗日函数：$^7$

> 7 在这里，非常重要的一点是要注意到全体$(w,b,\xi)$在“$x$”原变量中占据的角色。类似的，要注意到全体$(\alpha,\beta)$在“$\alpha$”对偶变量中占据的角色，通常是用于不等式约束的。因为在本问题中并没有仿射不等式约束，因此在这里就没有“$\beta$”对偶变量。

$$
\mathcal{L}(w,b,\xi,\alpha,\beta)=\frac 12 \parallel w\parallel^2+C\sum^m_{i=1}\xi_i+\sum^m_{i=1}\alpha_i(1-\xi_i-y^{(i)}(w^Tx^{(i)}+b))-\sum_{i=1}^m\beta_i\xi_i
$$

上式给出了原始和对偶优化问题：

$$
\begin{aligned}
\max_{\alpha,\beta:\alpha_i\ge0,\beta_i\ge0}\theta_{\mathcal{D}}(\alpha,\beta) \qquad&其中\qquad\theta_{\mathcal{D}}(\alpha,\beta) := \min_{w,b,\xi}\mathcal{L}(w,b,\xi,\alpha,\beta),\quad&(SVM-D)\\
\min_{w,b,\xi}\theta_{\mathcal{P}}(w,b,\xi)\qquad&其中\qquad\theta_{\mathcal{P}}(w,b,\xi) := \max_{\alpha,\beta:\alpha_i\ge0,\beta_i\ge0}\mathcal{L}(w,b,\xi,\alpha,\beta),\quad&(SVM-P)
\end{aligned}
$$

不过，要把对偶问题化成讲义中所示的形式，我们还有一些工作要做。特别是,

1. **消去原始变量。** 为了消除对偶问题中的原始变量，通过下面的式子计算$\theta_{\mathcal{D}}(\alpha,\beta)$：

$$
\theta_{\mathcal{D}}(\alpha,\beta)=\min_{w,b,\xi}\quad \mathcal{L}(w,b,\xi,\alpha,\beta)
$$

上式是一个无约束优化问题，其中目标函数$\mathcal{L}(w,b,\xi,\alpha,\beta)$是可微的。拉格朗日函数是关于$w$的严格凸二次函数，到目前为止，对于任意给定的$(\alpha,\beta)$来说，如果有$(\hat{w},\hat{b},\hat{\xi})$使得拉格朗日函数最小化，必须满足下式：

$$
\nabla_w\mathcal{L}(\hat{w},\hat{b},\hat{\xi},\alpha,\beta) = \hat{w}-\sum_{i=1}^m\alpha_iy^{(i)}x^{(i)}=0\qquad\qquad(24)
$$

跟进一步来说，拉格朗日函数是关于$b$和$\xi$的线性函数。通过类似于前一节简单对偶例子中描述的推理。我们可以设置对$b$和$\xi$求导等于零，并将得到的条件作为显式约束添加到对偶优化问题中：

$$
\frac{\partial}{\partial_b}\mathcal{L}(\hat{w},\hat{b},\hat{\xi},\alpha,\beta)=-\sum_{i=1}^m\alpha_iy^{(i)}=0\qquad\qquad(25) \\
\frac{\partial}{\partial_{\xi_i}}\mathcal{L}(\hat{w},\hat{b},\hat{\xi},\alpha,\beta)=C-\alpha_i-\beta_i=0\qquad\qquad(26)
$$

我们可以用这些条件来计算对偶目标为：

$$
\begin{aligned}
\theta_{\mathcal{D}}(\alpha,\beta) &= \mathcal{L}(\hat{w},\hat{b},\hat{\xi})\\
&= \frac 12 \parallel \hat{w}\parallel^2+C\sum^m_{i=1}\hat{\xi_i}+\sum^m_{i=1}\alpha_i(1-\hat{\xi}_i-y^{(i)}(\hat{w}^Tx^{(i)}+\hat{b}))-\sum_{i=1}^m\beta_i\hat{\xi_i} \\
&= \frac 12 \parallel \hat{w}\parallel^2+C\sum^m_{i=1}\hat{\xi_i}+\sum^m_{i=1}\alpha_i(1-\hat{\xi}_i-y^{(i)}(\hat{w}^Tx^{(i)}))-\sum_{i=1}^m\beta_i\hat{\xi_i} \\
&= \frac 12 \parallel \hat{w}\parallel^2 + \sum^m_{i=1}\alpha_i(1-y^{(i)}(\hat{w}^Tx^{(i)}))
\end{aligned}
$$

其中第一个等式来自于给定$(\alpha,\beta)$最优的$(\hat{w},\hat{b},\hat{\xi})$，第二个等式使用广义拉格朗日函数的定义，第三个等式和第四个等式分别来自$(25)$和$(26)$。最后，使用$(24)$，可得：

$$
\begin{aligned}
\frac 12 \parallel \hat{w}\parallel^2 + \sum^m_{i=1}\alpha_i(1-y^{(i)}(\hat{w}^Tx^{(i)})) &= \sum^m_{i=1}\alpha_i + \frac 12 \parallel \hat{w}\parallel^2 - \hat{w}^T\sum^m_{i=1}\alpha_iy^{(i)}x^{(i)} \\
&= \sum^m_{i=1}\alpha_i + \frac 12 \parallel \hat{w}\parallel^2 - \parallel \hat{w}\parallel^2 \\
&= \sum^m_{i=1}\alpha_i - \frac 12 \parallel \hat{w}\parallel^2 \\
&= \sum^m_{i=1}\alpha_i - \frac 12\sum^m_{i=1}\sum^m_{j=1}\alpha_i\alpha_iy^{(i)}y^{(j)}<x^{(i)},x^{(j)}>
\end{aligned}
$$

因此，我们的对偶问题（因为没有更多原始变量和所有的显式约束）就很简单了：

$$
\begin{aligned}
\min_{\alpha,\beta} \quad & \sum^m_{i=1}\alpha_i - \frac 12\sum^m_{i=1}\sum^m_{j=1}\alpha_i\alpha_iy^{(i)}y^{(j)}<x^{(i)},x^{(j)}> \\
subject\quad to \quad&\alpha_i \geq 0,\quad &i=1,...,m\\
& \beta_i \geq 0, &i=1,...,m \\
& \alpha_i + \beta_i = C, &i=1,...,m \\
&\sum^m_{i=1}\alpha_iy^{(i)}=0
\end{aligned}
$$

2. **KKT互补性。** KKT互补要求对任何原始最优解$(w^*,b^*,\xi^*)$和对偶最优解$(\alpha^*,\beta^*)$都满足： 

$$
\begin{aligned}
\alpha_i^*(1-\xi_i^*-y^{(i)}(w^{*T}x^{(i)}+b^*)) &= 0 \\
\beta_i^*\xi_i^* &= 0
\end{aligned}
$$

对于$i = 1,\dots,m$。根据第一个条件我们可以得出如果$\alpha_i^*>0$那么为了使乘积为零，则$1-\xi_i^*-y^{(i)}(w^{*T}x^{(i)}+b^*)=0$。由此断定：

$$
y^{(i)}(w^{*T}x^{(i)}+b^*)\le 1
$$

根据原可行性有$\xi^*\ge 0$，类似的，如果$\beta_i^*>0$，则需要$\xi_i^*=0$来确保互补性。根据原约束条件$y^{(i)}(w^{*T}x^{(i)}+b^*)\ge 1-\xi_i$，可以得出：

$$
y^{(i)}(w^{*T}x^{(i)}+b^*)\ge 1
$$

最后，因为$\beta_i^*>0$等价于$\alpha_i^*<C$（因为$\alpha^* + \beta_i^* = C$），我们可以将KKT条件总结为如下式子：

$$
\alpha_i^*<C\quad\Rightarrow\quad y^{(i)}(w^{*T}x^{(i)}+b^*)\ge 1, \\
\alpha_i^*>0\quad\Rightarrow\quad y^{(i)}(w^{*T}x^{(i)}+b^*)\le 1
$$

或者等价的表示为：

$$
\alpha_i^*=0\quad\Rightarrow\quad y^{(i)}(w^{*T}x^{(i)}+b^*)\ge 1, \\
0<\alpha_i^*<C\quad\Rightarrow\quad y^{(i)}(w^{*T}x^{(i)}+b^*)= 1, \\
\alpha_i^*=C\quad\Rightarrow\quad y^{(i)}(w^{*T}x^{(i)}+b^*)\le 1
$$

3. **简化。** 通过观察下面数学形式的每一对约束，我们可以稍微整理一下对偶问题：

$$
\beta_i\ge 0\qquad\qquad \alpha_i + \beta_i = C
$$

上面的式子等价于一个单约束$\alpha_i\le C$；也就是说，如果我们相约解决下面的约束问题：

$$
\begin{aligned}
\min_{\alpha,\beta} \quad & \sum^m_{i=1}\alpha_i - \frac 12\sum^m_{i=1}\sum^m_{j=1}\alpha_i\alpha_iy^{(i)}y^{(j)}<x^{(i)},x^{(j)}> \\
subject\quad to \quad&0\le\alpha_i \le C,\quad &i=1,...,m,\qquad\qquad(27)\\
&\sum^m_{i=1}\alpha_iy^{(i)}=0
\end{aligned}
$$

并且随后设置$\beta_i=C-\alpha_i$，则$(\alpha,\beta)$对于前面的对偶问题是最优的。最后一种形式实际上是课堂讲稿中给出的软边界SVM对偶形式。

#### 4 后续研究方向

在许多实际任务中，$90\%$的挑战都涉及如何以凸优化的形式来改写优化问题。一旦找到了正确的形式，就可以使用软件包。因为许多已有的用于凸优化的软件包已经进行了很好的调优，以处理不同类型的优化问题。下面是一小部分可用的软件包：

- 商业包：CPLEX, MOSEK
- 基于matlab的包：CVX，优化工具箱(linprog, quadprog)， SeDuMi
- 库：CVXOPT (Python)、GLPK (C)、COIN-OR (C)
- 支持向量机：LIBSVM SVM-light
- 机器学习：Weka (Java)

我们特别指出CVX作为一个易于使用的通用工具可以基于MATLAB求解凸优化问题，还有CVXOPT作为一个强大的基于Python库，其独立于MATLAB运行。$^8$ 如果你对上诉列表中出现的其他包感兴趣的话，可以很容易的在web中搜索到。简而言之，如果你需要一个特定的凸优化算法，现有的软件包提供了一种快速的原型化方法来让你实现该算法，而无需你自己完整的完成凸优化的所有数值计算。

>8 CVX 在网址 http://cvxr.com/cvx/ 可以找到。CVXOPT 在网址 http://cvxopt.org/ 可以找到。

另外，如果你觉得本材料很有趣，一定要看看Stephen Boyd的课程EE364: Optimization I，它将在冬季学期提供。EE364的课程教材（在参考资料[1]`注：参考资料[1]见文章最下方`中列出）包含丰富的凸优化知识，可以在线浏览。

##### 参考资料

<blockquote id='[1]'>[1] Stephen Boyd and Lieven Vandenberghe. Convex Optimization. Cambridge UP, 2004. Online: <a target='_blank' href='http://www.stanford.edu/~boyd/cvxbook/'>http://www.stanford.edu/~boyd/cvxbook/</a></blockquote>