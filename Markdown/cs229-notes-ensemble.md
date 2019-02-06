# CS229 课程讲义中文翻译
CS229 Lecture notes

|原作者|翻译|
|---|---|
| Raphael John Lamarre Townshend|[CycleUser](https://www.zhihu.com/people/cycleuser/columns)|


|相关链接|
|---|
|[Github 地址](https://github.com/Kivy-CN/Stanford-CS-229-CN)|
|[知乎专栏](https://zhuanlan.zhihu.com/MachineLearn)|
|[斯坦福大学 CS229 课程网站](http://cs229.stanford.edu/)|
|[网易公开课中文字幕视频](http://open.163.com/movie/2008/1/M/C/M6SGF6VB4_M6SGHFBMC.html)|

# 集成学习(Ensembling Methods)

现在要讲的方法可以来整合训练模型的输出.这里要用到偏差-方差(Bias-Variance)分析,以及决策树的样本来探讨一下每一种方法所做的妥协权衡.

要理解为什么从继承方法推导收益函数(benefit),首先会议一些基本的概率论内容.加入我们有n个独立同分布(independent, identically distributed,缩写为i.i.d.) 的随机变量$X_i$,其中的$0\le i<n$.假如对所有的$X_i$有$Var(X_i)=\sigma^2$.然后就可以得到均值(mean)的方差(variance)是:

$$
Var(\bar x)=Var(\frac{1}{n}\sum_iX_i)=\frac{\sigma^2}{n}
$$

现在就去掉了独立性假设(所以变量就只是同分布的,即i.d.),然后我们说变量$X_i$是通过一个因子(factor)$\rho$而相关联的,可以得到:

$$
\begin{aligned}
   Var(\bar X) &= Var (\frac{1}{n}\sum_iX_i) & \quad\text{(1)}\\
    &= \frac{1}{n^2}\sum_{ij}Cov(X_i,X_j)  & \quad\text{(2)}\\
    &= \frac{n\sigma^2}{n^2} +\frac{n(n-1)\rho\sigma^2}{n^2}  & \quad\text{(3)}\\
    &=  \rho\sigma^2+ \frac{1-\rho}{n}\sigma^2&  \quad\text{(4)}\\
\end{aligned}
$$

在第3步中用到了皮尔逊相关系数(pearson correlation coefficient)的定义$P_{X,Y}=\frac{Cov(X,Y)}{\sigma_x\sigma_y}$,而其中的$Cov(X,X)=Var(X)$.

现在如果我们设想每个随机变量都是一个给定模型的误差,就可以看到用到的模型的数目在增长(导致第二项消失),另外模型间的相关性在降低(使得第一项消失并且得到一个独立同分布的定义),最终导致了集成方法误差的方差的总体降低.

有好几种方法能够生成去除先关的模型(de-correlated models),包括:

* 使用不同算法
* 使用不同训练集
* 自助聚合(Bagging)
* Boosting

前两个方法很直接,就是需要大规模的额外工作.接下来讲一下后两种技术,boosting和bagging,以及在决策树情境下这两者的使用.

## 1 Bagging

### 1.1 自助 (Bootstrap)

Bagging这个词的意思是"Boostrap Aggregation"的缩写,是一个方差降低(variance reduction)的集成学习方法(ensembling method).Bootstrap这个方法是传统统计中用于测量某个估计器(比如均值mean)的不确定性的.



### 1.2 聚合 (Aggregation)

### 1.3 自助聚合 + 决策树 (Bagging+Decision Trees)

### 1.4 本节概要

## 2 Boosting

### 2.1 直观理解 (Intuition)

### 2.2 Adaboost 算法

### 2.3 正向累加建模(Forward Stagewise Additive Modeling)

### 2.4 梯度Boosting (Gradient Boosting)

### 2.5 本节概要