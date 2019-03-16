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

在上一篇的笔记中，我们开始学习**凸优化，** 并且学习到了如下数学优化问题的形式：

$$
\begin{aligned}
\min_{x\in R^n}\quad&f(x) \\
subject\quad to\quad &x\in C \qquad\qquad (1)
\end{aligned}
$$

在凸优化问题的设定中，$x\in R^n$是一个被称作**优化变量**的向量。$f:R^n\rightarrow R$是我们想要优化的**凸函数。** $c\subseteq R^n$是一个被称作可行域集合的**凸集。** 从计算的角度来看，凸优化问题的有趣之处在于，任何局部最优解总是保证其也是全局最优的解。在过去的几十年里，求解凸优化问题的通用方法也变得越来越可靠有效。

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

上面的等式中，函数$\theta_\mathcal{P}:R^n\rightarrow R$被称作**原目标(primal objective,)，** 右边的无约束极小化问题称为**原问题(primal problem)。** 在通常情况下，当$g_i(x)\le 0,i=1,\dots,m$以及$h_i(x)=0,i=1,\dots,p$时，一个点$x\in R^n$被称作**原可行域(primal feasible)。** 我们通常使用向量$x^*\in R^n$代表$(P)$式的解。我们令$p^*=\theta_\mathcal{P}(x^*)$代表原目标的优化值。

<u>对偶问题</u>

通过对上述最小化和最大化顺序的转换，我们得到了一个完全不同的优化问题：

$$
\max_{\alpha,\beta:\alpha_i\ge0,\forall i}\underbrace{[\min_x\mathcal{L}(x,\alpha,\beta)]}_{这部分称作\theta_\mathcal{D}(\alpha,\beta)}=\max_{\alpha,\beta:\alpha_i\ge 0,\forall i}\theta_\mathcal{D}(\alpha,\beta)\qquad\qquad(D)
$$

这里的函数$\theta_\mathcal{D}:R^m\times R^p\rightarrow R$被称作**对偶目标(dual objective)，** 右边的约束最大化问题称为**对偶问题(dual problem)。** 通常情况下，当$\alpha_i\ge 0,i=1,\dots,m$时，我们称$(\alpha,\beta)$为**对偶可行域(dual feasible)。** 我们通常使用向量对$(\alpha^*,\beta^*)\in R^m\times R^p$代表$(D)$式的解。我们令$\theta_\mathcal{D}(\alpha^*,\beta^*)$代表对偶目标的优化值。

##### 1.3 原问题的解释

首先观察到，原目标函数$\theta_\mathcal{P}(x)$是一个关于$x$的凸函数$^3$。为了解释原问题，注意到：

<blockquote><details><summary>3 原目标函数是凸函数的原因</summary>
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

- 如果任意$g_i(x)>0$，则使括号内表达式最大化需要使对应的$\alpha_i$为任意大的正数；但是，如果$g_i(x)\le 0$，且需要$\alpha_i$非负，这就意味着调节$\alpha_i$达到整体最大值的设置为$\alpha_i= 0$，此时的最大值为0。

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
\end{cases}
$$

作为一种障碍函数(“barrier” function)，它防止我们将不可行点作为优化问题的候选解。

##### 1.4 对偶问题的解释

对偶目标函数$\theta_\mathcal{D}(\alpha,\beta)$是一个关于$\alpha$和$\beta$的凸函数$^4$。为了解释对偶问题，我们首先做如下观察:

<blockquote><details><summary>4 对偶目标函数是凸函数的原因</summary>
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

这里，第一步和第三步分别直接遵循对偶目标函数和拉格朗日函数的定义。第二步是根据前面的表达式在$x$的可能值上取得最小化，最后一步是根据$x^*$是在原可行域内的这个事实得出的。因此等式$(8)$暗示等式$(12)$的后两项必须是非正数。

引理表明，给定任何对偶可行的$(\alpha,\beta)$，对偶目标$\theta_\mathcal{D}(\alpha,\beta)$提供了原问题优化值$p^*$的一个下界。由于对偶问题涉及到在所有对偶可行域$(\alpha,\beta)$上使对偶目标最大化。因此，对偶问题可以看作是对可能的最紧下界$p^*$的搜索。这就对任何原始和对偶优化问题对产生了一个性质，这个性质被称为**弱对偶(weak duality)：**

**引理 2** （弱对偶）。对于任何一对原始的和对偶的问题，$d^*\le p^*$。

显然，弱对偶性是引理$1$使用$(\alpha,\beta)$作为对偶可行点的结果。对于一些原始/对偶优化问题，一个更强的可以成立的结果，其被称为**强对偶性(strong duality)：**

**引理 3** （强对偶）。对于满足一定被称为**约束规定(constraint qualifications)** 的技术条件的任何一对原问题和对偶问题， 可得到$d^*= p^*$。

存在许多不同的约束条件，其中最常调用的约束条件称为**Slater条件(Slater’s condition)：** 如果存在一个所有不等式约束（即$g_i(x)<0,i=1,\dots,m$）都严格满足的可行原始解$x$，则原始/对偶问题对满足Slater条件。在实际应用中，几乎所有的凸问题都满足某种约束条件，因此原问题和对偶问题具有相同的最优值。

##### 1.5 互补松弛性 

凸优化问题强对偶性的一个特别有趣的结果是**互补松弛性(complementary slackness)**（或KKT互补）：

**引理 4** （互补松弛性）。如果强对偶成立，则对于每一个$i=1,\dots,m$都有$\alpha_i^*g(x_i^*)$

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

但是，回忆一下每个$\alpha_i^*$是非负的，每个$g_i(x^*)$是非负的，并且由于$x^*$和$(\alpha^*,\beta^*)$分别是原可行域和对偶可行域，所以每个$h_i(x^*)$都是零。因此，$(18)$表示的是所有非正项之和等于零的一个式子。很容易得出结论，求和中的所有单独项本身都必须为零（因为如果不为零，求和中就没有允许总体和保持为零的补偿正项）。

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

定理 1.1 假设$x^*\in R^n,\alpha^*\in R^m,\beta^*\in R^p$满足以下条件：

1. （原始的可行性）$g_i(x^*)\le 0,i=1,\dots,m$以及$h_i(x^*)=0,i=1,\dots,p$，
2. （对偶可行性）$\alpha_i^*\ge 0,i= 1,\dots,m$，
3. （互补松弛性）$\alpha_i^*g_i(x^*)=0,i= 1,\dots,m$，
4. （拉格朗日平稳性）$\nabla_x\mathcal{L}(x^*,\alpha^*,\beta^*)=0$。

$x^*$是原优化，$(\alpha^*,\beta^*)$是对偶优化。更进一步，如果强对偶成立，则任意原优化$x^*$以及对偶优化$(\alpha^*,\beta^*)$必须满足条件$1$到条件$4$。

这些条件被称为Karush-Kuhn-Tucker (KKT)条件。$^5$

>5 顺便提一下，KKT定理有一段有趣的历史。这个结果最初是由卡鲁什在1939年的硕士论文中推导出来的，但直到1950年被两位数学家库恩和塔克重新发现，才引起人们的注意。约翰在1948年也推导出了本质上相同结果的一个变体。关于为什么这个结果的如此多的迭代在近十年中被忽视的有趣的历史解释，请看这篇论文：
Kjeldsen, T.H. (2000) A contextualized historical analysis of the Kuhn-Tucker Theorem in nonlinear programming: the impact of World War II. Historica Mathematics 27: 331-361.

