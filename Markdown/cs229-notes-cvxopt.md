# CS229 课程讲义中文翻译
CS229 Section notes

|原作者|翻译|
|---|---|
|Zico Kolter|[XiaoDong_Wang](https://github.com/Dongzhixiao) |


|相关链接|
|---|
|[Github 地址](https://github。com/Kivy-CN/Stanford-CS-229-CN)|
|[知乎专栏](https://zhuanlan。zhihu。com/MachineLearn)|
|[斯坦福大学 CS229 课程网站](http://cs229。stanford。edu/)|
|[网易公开课中文字幕视频](http://open。163。com/movie/2008/1/M/C/M6SGF6VB4_M6SGHFBMC。html)|


### 凸优化概述

#### 1. 介绍

在很多时候，我们进行机器学习算法时希望优化某些函数的值。即，给定一个函数$f:R^n\rightarrow R$，我们想求出使函数$f(x)$最小化（或最大化）的原像$x\in R^n$。我们已经看过几个包含优化问题的机器学习算法的例子，如：最小二乘算法、逻辑回归算法和支持向量机算法，它们都可以构造出优化问题。

在一般情况下，很多案例的结果表明，想要找到一个函数的全局最优值是一项非常困难的任务。然而，对于一类特殊的优化问题——**凸优化问题，** 我们可以在很多情况下有效地找到全局最优解。在这里，有效率既有实际意义，也有理论意义：它意味着我们可以在合理的时间内解决任何现实世界的问题，它意味着理论上我们可以在一定的时间内解决该问题，而时间的多少只取决于问题的多项式大小。**（译者注：即算法的时间复杂度多项式时间级别$O(n^k)$，其中$k$代表多项式中的最高次数）**

这部分笔记和随附课程的目的是对凸优化领域做一个非常简要的概述。这里的大部分材料（包括一些数字）都是基于斯蒂芬·博伊德(Stephen Boyd)和利文·范登伯格(lieven Vandenberghe)的著作《凸优化》（凸优化<a target='_blank' href='https://web.stanford.edu/~boyd/cvxbook/'>[1]</a>在网上<a target='_blank' href='https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf'>免费提供下载</a>）以及斯蒂芬·博伊德(Stephen Boyd)在斯坦福教授的课程EE364。如果您对进一步研究凸优化感兴趣，这两种方法都是很好的资源。

