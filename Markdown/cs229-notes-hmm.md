# CS229 课程讲义中文翻译
CS229 Section notes

|原作者|翻译|
|---|---|
|Daniel Ramage|[XiaoDong_Wang](https://github.com/Dongzhixiao) |


|相关链接|
|---|
|[Github 地址](https://github。com/Kivy-CN/Stanford-CS-229-CN)|
|[知乎专栏](https://zhuanlan。zhihu。com/MachineLearn)|
|[斯坦福大学 CS229 课程网站](http://cs229。stanford。edu/)|
|[网易公开课中文字幕视频](http://open。163。com/movie/2008/1/M/C/M6SGF6VB4_M6SGHFBMC。html)|


### 隐马尔可夫模型基础

#### 摘要

我们如何将机器学习应用于随时间变化观察到的一系列数据中来？例如，我们可能对根据一个人讲话的录音来发现他所说的话的顺序感兴趣。或者，我们可能对用词性标记来注释单词序列感兴趣。本小节的内容对马尔可夫模型的概念进行了全面的数学介绍，该模型是一种关于状态随时间变化的推理一种学习形式。并且使用隐马尔可夫模型，我们希望从一系列观察数据中恢复这一系列模型的初始状态。最后一节包含一些特定参考资料，这些资料从其他角度介绍隐马尔可夫模型。

#### 1. 马尔科夫模型

给定一个状态集合$S=\{s_1,s_2,\dots,s_{|s|}\}$，我们可以观察到一系列随时间变化的序列$\vec{z}\in S^T$。例如，我们也许有这样一个来自天气系统的状态集合$S=\{sun,cloud,rain\}$，显然$|S|=3$。在给定$T=5$的情况下我们可能会观察到这几天的天气情况的一个序列$\{z_1=s_{sun},z_2=s_{cloud},z_3=s_{cloud},z_4=s_{rain},z_5=s_{cloud}\}$

我们上面的天气示例里面的观察状态可以表示随时间变化的一种随机过程的输出。如果没有进一步的假设，时间$t$下的状态$s_j$可以是自变量为任意数的一个函数，包括从时间$1$到$t-1$的所有状态，可能还有许多其它我们甚至没有建模的状态。然而，我们将做两个马尔可夫假设，这将允许我们对时间序列进行可以追溯的推断。

有限地平线假设(limited horizon assumption)是$t$时刻处于状态的概率只取决于$t-1$时刻的状态。这个假设背后的直觉是，$t$时刻的状态代表对过去“足够”的总结，可以合理地预测未来。正式的公式如下:

$$
P(z_t|z_{t-1},z_{t-2},\dots,z_1)=P(z_t|z_{t-1})
$$

平稳过程假设(stationary process assumption)是在给定当前状态的条件下，下一个状态的条件分布不随时间变化。正式的公式如下:

$$
P(z_t|z_{t-1})=P(z_2|z_1);t\in 2\dots T
$$

习惯上，我们还将假设存在一个初始状态和初始观察值$z_0\equiv s_0$，其中$s_0$为$0$时刻状态的初始概率分布。这种符号定义可以使我们方便编码观察到第一个真实的状态$z_1$的先验概率的确信度，其可以用符号表示为$p(z_1|z_0)$。注意到公式$P(z_t|z_{t-1},\dots,z_1)=P(z_t|z_{t-1},\dots,z_1,z_0)$成立是因为我们为所有状态序列都定义了$z_0=s_0$。（HMMs的其它表示形式有时用向量$\pi\in R^{|S|}$表示这些先验确信度(prior believes)）

我们通过定义一个状态转移矩阵$A\in R^{(|S|+1)\times(|S|+1)}$来参数化这些转移数据。矩阵中的值$A_{ij}$代表在任意时刻$t$从状态$i$转移到状态$j$的转移概率。对于我们太阳和雨的例子，可能有下面的状态转移矩阵：

$$
A=\begin{matrix}
\ & s_0&s_{sun} & s_{cloud} & s_{rain}\\
s_0 &  0 & .33 & .33 & .33 \\
s_{sun} & 0 & .8 & .1 & .1 \\
s_{cloud} & 0 & .2 & .6 & .2\\
s_{rain} & 0 & .1 & .2 & .7
\end{matrix}
$$

请注意，这些数字（我自己编的）表明了天气是自相关的，这是因为：如果天气晴朗，它将趋向于保持晴朗，如果天气多云将保持多云等等。这种模式在许多马尔可夫模型中都很常见，可以作为转移矩阵中的强对角性来遵守。注意，在本例中，我们的初始状态$s_0$显示了过渡到天气系统中的三种状态的概率是一样的。

##### 1.1 马尔可夫模型的两个问题

结合马尔可夫假设和状态转移参数矩阵$A$，我们可以回答关于马尔可夫链中状态序列的两个基本问题。
- 给定一个特定的状态序列$\vec{z}$，其概率是多少？
- 给定一个观测序列$\vec{z}$，如何通过其进行最大似然估计得到状态转移参数矩阵$A$？

###### 1.1.1 状态序列的概率

我们可以利用概率的链式法则来计算某一特定状态序列$\vec{z}$的概率：

$$
\begin{aligned}
P（\vec{z}) &= P(z_t,z_{t-1},\dots,z_1;A) \\
&= P(z_t,z_{t-1},\dots,z_1,z_0;A) \\
&= P(z_t|z_{t-1},z_{t-2},\dots,z_1;A)P(z_{t-1}|z_{t-2},\dots,z_1;A)\dots P(z_1|z_0;A) \\
&= P(z_t|z_{t-1};A)P(z_{t-1}|z_{t-2};A)\dots P(z_2|z_1;A)P(z_1|z_0;A) \\
&= \prod_{t=1}^TP(z_t|z_{t-1};A) \\
&= \prod_{t=1}^TA_{z_{t-1} z_t}
\end{aligned}
$$

在第二行，我们在联合概率密度的公式中引入$z_0$，这使得该式可以通过前面定义的$z_0$来计算。第三行的结果是通过将概率链式法则或贝叶斯规则的重复应用到该联合概率密度上得到的。第四行遵循马尔可夫假设，最后一行表明这些项都来自于状态转换矩阵$A$中的元素。

我们计算一下前面例子中的时间序列的概率。通过式子表达的话，即我们想要计算$P(z_1 = s_{sun} , z_2 = s_{cloud} , z_3 = s_{rain} , z_4 = s_{rain} , z_5 = s_{cloud})$，这个式子可以通过分解来计算，即$P(s_{sun}|s_0)P(s_{cloud}|s_{sun})P(s_{rain}|s_{cloud})P(s_{rain}|s_{rain})P(s_{cloud}|s_{rain}) =.33 \times .1 \times .2 \times .7 \times .2$。

###### 1.1.2 最大似然参数赋值

从学习的角度来看，我们可以通过观察序列$\vec{z}$的对数似然函数找到参数矩阵$A$。相应的找到从晴天到多云或者从晴天到晴天等转移的似然，最大化以使得观察集合发生的概率最大。让我们定义一个马尔科夫模型的对数似然函数：

$$
\begin{aligned}
l(A) &= logP(\vec{z};A) \\
&= log\prod_{t=1}^TA_{z_{t-1} z_t} \\
&= \sum_{t=1}^TlogA_{z_{t-1} z_t} \\
&= \sum_{i=1}^{|S|}\sum_{j=1}^{|S|}\sum_{t=1}^{T}1\{z_{t-1}=s_i\wedge z_t=s_j\}logA_{ij}
\end{aligned}
$$

在最后一行中，我们使用一个示性函数，当大括号内的条件满足时，它的值为$1$，否则为$0$，通过该函数在每个时间步长选择观察到的转换。在求解这一优化问题时，重要的是要保证所求解的参数矩阵$A$仍然是一个有效的转移矩阵。特别地，我们需要确保状态$i$的输出概率分布总是和为$1$，并且$A$的所有元素都是非负的。我们可以用拉格朗日乘子法来求解这个优化问题。

$$
\begin{aligned}
\max_A\qquad &l(A) \\
s.t.\qquad &\sum_{j=1}^{|S|}A_{ij}=1,\quad i=1..|S|\\
&A_{ij}\ge 0,\quad i,j=1..|S|
\end{aligned}
$$

该约束优化问题可以用拉格朗日乘子法求得闭式解。我们将把等式约束带入拉格朗日方程，但不等式约束可以放心地忽略——因为优化解总能为$A_{ij}$产生一个正值。因此我们构建如下的拉格朗日函数：

$$
\mathcal{L}(A,\alpha)=\sum_{i=1}^{|S|}\sum_{j=1}^{|S|}\sum_{t=1}^{T}1\{z_{t-1}=s_i\wedge z_t=s_j\}logA_{ij}+\sum_{i=1}^{|S|}\alpha_i(1-\sum_{j=1}^{|S|}A_{ij})
$$

求偏导数，令它们等于零可得:

$$
\begin{aligned}
\frac{\partial\mathcal{L}(A,\alpha)}{\partial A_{ij}} &=\frac{\partial}{\partial A_{ij}}(\sum_{t=1}^{T}1\{z_{t-1}=s_i\wedge z_t=s_j\}logA_{ij}) + \frac{\partial}{\partial A_{ij}}\alpha_i(1-\sum_{j=1}^{|S|}A_{ij}) \\
&= \frac 1{A_{ij}}\sum_{t=1}^{T}1\{z_{t-1}=s_i\wedge z_t=s_j\}-\alpha_i\equiv0\\
&\Rightarrow \\
A_{ij} &=\frac 1{\alpha_i}\sum_{t=1}^{T}1\{z_{t-1}=s_i\wedge z_t=s_j\}
\end{aligned}
$$

回带原式，并令其对于$\alpha$的偏导等于零可得：

$$
\begin{aligned}
\frac{\partial\mathcal{L}(A,\alpha)}{\partial \alpha_i} &= 1-\sum_{j=1}^{|S|}A_{ij} \\
&= 1-\sum_{j=1}^{|S|}\frac 1{\alpha_i}\sum_{t=1}^{T}1\{z_{t-1}=s_i\wedge z_t=s_j\}\equiv0 \\
&\Rightarrow \\
\alpha_i &= \sum_{j=1}^{|S|}\sum_{t=1}^{T}1\{z_{t-1}=s_i\wedge z_t=s_j\} \\
&= \sum_{t=1}^{T}1\{z_{t-1}=s_i\}
\end{aligned}
$$

把$\alpha_i$的值带入相应表达式，我们推导出$A_{ij}$的最大似然参数值$\hat{A_{ij}}$为：

$$
\hat{A_{ij}} = \frac{\sum_{t=1}^T 1\{z_{t-1}=s_i\wedge z_t=s_j\}}{\sum_{t=1}^T 1\{z_{t-1} = s_i\}}
$$

这个公式结果表达的一个简单的解释是：从状态$i$到状态$j$转移的最大似然概率其实就是从状态$i$到状态$j$出现的次数数除以总次数。换句话说，就是最大似然参数等于我们从状态$i$到状态$j$的次数比上我们在状态$i$中的次数的分数。

#### 2. 隐马尔科夫模型

马尔可夫模型是对时间序列数据的一种强大抽象，但无法捕获非常常见的场景。如果我们不能观察状态本身，而只能观察这些状态的一些概率函数，我们怎么能对一系列状态进行推理呢？比如一个词性标注的场景，其中单词被观察到，但是词性标记没有被观察到。或者在语音识别的场景中，语音序列被观察到，但是生成它的单词没有被观察到。举个简单的例子，让我们借用Jason Eisner在2002[1]`参考资料[1]见文章最下方`年提出的设置，即“冰淇淋气候学”：

情境：在2799年，你是一位气候学家，研究全球变暖的历史。你找不到巴尔的摩(Baltimore)天气的任何记录，但你找到了我（杰森·艾斯纳(Jason Eisner)）的日记。我勤奋地记录我每天吃了多少冰淇淋。关于那个夏天的天气情况，你能推断出什么？

可以使用隐马尔可夫模型(HMM)来研究这个场景。我们不能观察状态的实际序列（每天天气情况的序列）。相反，我们只能观察每个天气状态产生的一些结果（那天吃了多少冰淇淋）。

形式上，HMM是一个马尔可夫模型，我们有一系列观察到的输出$x=\{x_1,x_2,\dots,x_T\}$，该输出来自于一组输出符号集(an output alphabet)$V=\{v_1,v_2,\dots,v_{|V|}\}$，即$x_t\in V,t=1..T$。和上一节一样，我们也假定了一系列状态的存在，这些状态来自于一个状态符号集合$S=\{s_1,s_2,\dots s_{|s|}\},z_t\in S,t=1..T$，但是在这种情况下，状态值是不可见的。状态$i$和$j$之间的转换将再次用状态转移矩阵$A_{ij}$中的对应值表示。

我们还将生成输出观测值的概率作为隐状态的函数来建模。为此，我们做了输出无关的假设(output independence assumption)，同时定义$P(x_t=v_k|z_t=s_j)=P(x_t=v_k|x_1,\dots,x_T,z_1,\dots,z_T)=B_{jk}$。矩阵$B$编码了隐藏状态产生输出$v_k$的概率，$v_k$在相应时间产生的状态是$s_j$。

回到天气的例子，假设你有四天的冰淇淋消费记录$\vec{x}=\{x_1=v_3,x_2=v_2,x_3=v_1,x_4=v_2\}$。其中我们的观察集合仅仅有冰激凌消耗的数量，即$V=\{v_1=1冰激凌,v_2=2冰激凌,v_3=3冰激凌\}$。HMM能给我们回答什么问题呢？

##### 2.1 隐马尔科夫模型的三个问题

我们可能会问HMM三个基本问题。观察到的序列的概率是多少（比如我们观察到消耗了$3,2,3,2$个冰淇淋）？最有可能产生观测结果的一系列状态是什么（那四天的天气如何）？我们如何学习给定数据时的隐马尔可夫模型参数$A$和$B$的值？

##### 2.2 观测序列的概率：正演过程

在HMM中，我们假设数据是由以下过程生成的：假设存在一系列基于我们时间训序列长度的状态$\vec{z}$。该状态序列由状态转换矩阵$A$参数化的马尔可夫模型生成。在每个时间步$t$，我们选择一个输出$x_t$作为状态$z_t$出现下的函数。因此，为了得到一个观测序列的概率，我们需要将给定的每个可能状态序列的数据$\vec{x}$的似然概率相加。

$$
\begin{aligned}
P(\vec{x};A,B) &= \sum_{\vec{z}}P(\vec{x},\vec{z};A,B) \\
&= \sum_{\vec{z}}P(\vec{x}|\vec{z};A,B)P(\vec{z};A,B)
\end{aligned}
$$

上述公式适用于任何概率分布。然而，HMM假设允许我们进一步简化表达式：

$$
\begin{aligned}
P(\vec{x};A,B) &= \sum_{\vec{z}}P(\vec{x}|\vec{z};A,B)P(\vec{z};A,B) \\
&= \sum_{\vec{z}}(\prod_{t=1}^TP(x_t|z_t;B))(\prod_{t=1}^TP(z_t|z_{t-1};A)) \\
&= \sum_{\vec{z}}(\prod_{t=1}^TB_{z_tx_t})(\prod_{t=1}^TA_{z_{t-1}z_t})
\end{aligned}
$$

好消息是，上式是一个关于参数的简单表达式。推导过程遵循HMM假设：输出独立假设、马尔可夫假设和平稳过程假设，这三个假设都用于推导第二行。坏消息是所有可能的产生序列$\vec{z}$情况的总和太大了。因为$z_t$在每个时间步都可能有$|S|$种可能情况，直接计算总和需要操作的时间复杂度是$O(|S|^T)$。

<hr style="height:1px;border:none;border-top:3px solid black;" />

**算法 1** 前向算法计算$\alpha_i(t)$

<hr style="height:1px;border:none;border-top:1px solid black;" />

1. 基本情况：$\alpha_i(0) = A_{0i},i=1..|s|$

2.  递归： $\alpha_j(t) = \sum_{i=1}^{|S|}\alpha_i(t-1)A_{ij}B_{jx_t},j=1..|S|,t=1..T$

<hr style="height:1px;border:none;border-top:1px solid black;" />

幸运的是，可以根据一个名叫前向算法(Forward Procedure)的算法更快的计算$P(\vec{x};A,B)$，该算法采用了动态规划的思想。首先让我们定义一个符号：$\alpha_i(t)=P(x_1,x_1,\dots,x_t,z_t=s_i;A,B)$。$\alpha_i(t)$代表随时间$t$（通过任意状态指定）变化的所有观测值和我们在时间$t$进入状态$s_i$的联合概率。在我们有了这个符号之后，所有观察到对象的全集的概率$P(\vec{x})$可以如下表达：

$$
\begin{aligned}
P(\vec{x};A,B) &= P(x_1,x_2,\dots,x_T;A,B) \\
&= \sum_{i=1}^{|S|}P(x_1,x_2,\dots,x_T,z_T=s_i;A,B) \\
&= \sum_{i=1}^{|S|}\alpha_i(T)
\end{aligned}
$$

算法$1$给出了一种有效的方法来计算$\alpha_i(t)$。在每个时间步，我们进行计算的时间复杂度仅仅是$O(|S|)$，这样得到最终计算观察到的状态序列的总概率$P(\vec{x};A,B)$算法的时间复杂度是$O(|S|\times T)$。

一个类似称为向后过程(Backward Procedure)的算法可以用来计算类似的概率$\beta_i(t)=P(x_T,x_{T-1},\dots,x_{t+1},z_t=s_i;A,B)$。

##### 2.3 最大似然状态目标序列：维特比算法

隐马尔可夫模型最常见的问题之一是想要知道在给定了一个观察到的输出序列$\vec{x}\in V^T$时，最有可能的状态序列$\vec{z}\in S^T$是什么。可以用如下公式表达：

$$
arg\max_{\vec{z}}P(\vec{z}|\vec{x};A,B)=arg\max_{\vec{z}} \frac{P(\vec{x}, \vec{z};A,B)}{\sum_{\vec{z}}P(\vec{x}, \vec{z};A,B)}=arg\max_{\vec{z}}P(\vec{x}, \vec{z};A,B)
$$

第一个化简遵循贝叶斯规则，第二个化简遵循分母不直接依赖$\vec{z}$的观察结果。简而言之，我们这里模型的意思是尝试所有可能产生目标序列$\vec{z}$，并取其中能使得联合概率最大的那个目标序列。然而，枚举一组可能的任务序列需要的时间复杂度是$O(|S|^T)$。在这一点上，你可能会想到使用上一小节的正向算法那样的动态规划方案来解决本节的问题可能会节约时间，没错。注意，如果将$arg\max_{\vec{z}}$替换为$\sum_{\vec{z}}$，那么我们当前的任务与前向算法的表达式完全类似。

<hr style="height:1px;border:none;border-top:3px solid black;" />

**算法 2** 基于$EM$算法解决隐马尔可夫模型普通应用的算法： 

<hr style="height:1px;border:none;border-top:1px solid black;" />

（$E$步）对于每一个可能的序列$\vec{z} \in S^T$，设：

$$
Q(\vec{z}):=p(\vec{z}|\vec{x};A, B)
$$

（$M$步）设：

$$
\begin{aligned}
A, B &:= arg\max_{A,B}\sum_{\vec{z}}Q(\vec{z})log\frac{P(\vec{x}, \vec{z}; A, B)}{Q(\vec{z})} \\
&s.t.\sum_{j=1}^{|S|}A_{ij}=1,i=1...|S|;A_{ij}\ge0,\quad i,j=1...|S| \\
&\quad\sum_{k=1}^{|V|}B_{ik}=1,i=1...|S|;B_{ik}\ge0,\quad i=1...|S|,k=1...|V|
\end{aligned}
$$

<hr style="height:1px;border:none;border-top:1px solid black;" />

维特比算法(Viterbi Algorithm)与正向过程类似，不同之处在于，我们只需要跟踪最大概率并记录其对应的状态序列，而不是跟踪到目前为止所看到的生成观测结果的总概率。

##### 2.4 参数学习：基于EM算法的隐马尔可夫模型

HMM模型的最后一个问题是：给定一组观察序列的集合，使这组集合最有可能出现的状态转移概率矩阵(state transition probabilities)$A$和状态生成概率矩阵(output emission probabilities)$B$的值是多少？例如，基于语音识别数据集求解最大似然参数可以使我们有效地训练HMM模型，之后在需要求得候选语音信号的最大似然状态序列时使用该模型。

在本节中，我们推导了隐马尔可夫模型的期望最大化算法。这个证明来自于CS229课堂讲稿中给出的$EM$的一般公式。算法$2$给出了基本的$EM$算法。注意，$M$步中的优化问题现在受到约束，使得$A$和$B$包含有效的概率。就像我们为（非隐）马尔可夫模型找到的最大似然解一样，我们将能够用拉格朗日乘子来解决这个优化问题。还要注意，$E$步和$M$步都需要枚举所有$|S|^T$种可能的序列$\vec{z}$。我们将使用前面提到的前向和后向算法为我们的$E$步和$M$步计算一组有效的统计量。

首先，我们用马尔可夫假设重写目标函数：

$$
\begin{aligned}
A,B &= arg\max_{A,B}\sum_{\vec{z}}Q(\vec{z})log\frac{P(\vec{x},\vec{z};A,B)}{Q(\vec{z})} \\
&= arg\max_{A,B}\sum_{\vec{z}}Q(\vec{z})log P(\vec{x},\vec{z};A,B) \\
&= arg\max_{A,B}\sum_{\vec{z}}Q(\vec{z})log (\prod_{t=1}^TP(x_t|z_t;B))(\prod_{t=1}^TP(z_t|z_{t-1};A)) \\
&= arg\max_{A,B}\sum_{\vec{z}}Q(\vec{z})\sum_{t=1}^TlogB_{z_tx_t}+logA_{z_{t-1}z_t} \\
&= arg\max_{A,B}\sum_{\vec{z}}Q(\vec{z})\sum_{i=1}^{|S|}\sum_{j=1}^{|S|}\sum_{k=1}^{|V|}\sum_{t=1}^T1\{z_t=s_j\wedge x_t=v_k\}logB_{jk}+1\{z_{t-1}=s_i\wedge z_t=s_j\}logA_{ij}
\end{aligned}
$$

在第一行中，我们将对数除法分解为减法，注意分母的项不依赖于参数$A,B$。第$3$行应用了马尔可夫假设。第$5$行使用示性函数按状态索引$A$和$B$。

对于可见马尔可夫模型的最大似然参数，忽略不等式约束是安全的，因为解的形式自然只产生正解。构造拉格朗日函数：

$$
\begin{aligned}
\mathcal{L}(A,B,\delta,\epsilon) = &\sum_{\vec{z}}Q(\vec{z})\sum_{i=1}^{|S|}\sum_{j=1}^{|S|}\sum_{k=1}^{|V|}\sum_{t=1}^T1\{z_t=s_j\wedge x_t=v_k\}logB_{jk}+1\{z_{t-1}=s_i\wedge z_t=s_j\}logA_{ij}\\
&+ \sum_{j=1}^{|S|}\epsilon_j(1-\sum_{k=1}^{|V|}logB_{jk})+\sum_{i=1}^{|S|}\delta_i(1-\sum_{j=1}^{|S|}A_{ij})
\end{aligned}
$$

求偏导并使它们等于零：

$$
\begin{aligned}
\frac{\partial\mathcal{L}(A,B,\delta,\epsilon)}{\partial A_{ij}} &= \sum_{\vec{z}}Q(\vec{z})\frac 1{A_{ij}}\sum_{t=1}^T1\{z_{t-1}=s_i\wedge z_t=s_j\}-\delta_i\equiv 0 \\
A_{ij} &= \frac 1{\delta_i}\sum_{\vec{z}}Q(\vec{z})\sum_{t=1}^T1\{z_{t-1}=s_i\wedge z_t=s_j\} \\
\frac{\partial\mathcal{L}(A,B,\delta,\epsilon)}{\partial B_{jk}} &= \sum_{\vec{z}}Q(\vec{z})\frac 1{B_{jk}}\sum_{t=1}^T1\{z_t=s_j\wedge x_t=v_k\}-\epsilon_j\equiv 0 \\
B_{jk} &= \frac 1{\epsilon_j}\sum_{\vec{z}}Q(\vec{z})\sum_{t=1}^T1\{z_t=s_j\wedge x_t=v_k\}
\end{aligned}
$$

对拉格朗日乘子求导，代入上面$A_{ij}$和$B_{jk}$的值：

$$
\begin{aligned}
\frac{\partial\mathcal{L}(A,B,\delta,\epsilon)}{\partial \delta_i} &= 1 - \sum_{j=1}^{|S|}A_{ij} \\
&= 1 - \sum_{j=1}^{|S|}\frac 1{\delta_i}\sum_{\vec{z}}Q(\vec{z})\sum_{t=1}^T1\{z_{t-1}=s_i\wedge z_t=s_j\}\equiv 0 \\
\delta_i &= \sum_{j=1}^{|S|}\sum_{\vec{z}}Q(\vec{z})\sum_{t=1}^T1\{z_{t-1}=s_i\wedge z_t=s_j\} \\
&= \sum_{\vec{z}}Q(\vec{z})\sum_{t=1}^T1\{z_{t-1}=s_i\}  \\
\frac{\partial\mathcal{L}(A,B,\delta,\epsilon)}{\partial \epsilon_j} &= 1 - \sum_{k=1}^{|V|}B_{jk} \\
&= 1 - \sum_{k=1}^{|V|}\frac 1{\epsilon_j}\sum_{\vec{z}}Q(\vec{z})\sum_{t=1}^T1\{z_t=s_j\wedge x_t=v_k\}\equiv 0 \\
\epsilon_j &= \sum_{k=1}^{|V|}\sum_{\vec{z}}Q(\vec{z})\sum_{t=1}^T1\{z_t=s_j\wedge x_t=v_k\} \\
&= \sum_{\vec{z}}Q(\vec{z})\sum_{t=1}^T1\{z_t=s_j\}
\end{aligned}
$$

代回上面的表达式，我们得到参数$\hat{A}$和$\hat{B}$使我们对数据集的预测计数最大化：

$$
\begin{aligned}
\hat{A}_{ij} &= \frac{\sum_{\vec{z}}Q(\vec{z})\sum_{t=1}^T1\{z_{t-1}=s_i\wedge z_t=s_j\}}{\sum_{\vec{z}}Q(\vec{z})\sum_{t=1}^T1\{z_{t-1}=s_i\}} \\
\hat{B}_{jk} &= \frac{\sum_{\vec{z}}Q(\vec{z})\sum_{t=1}^T1\{z_t=s_j\wedge x_t=v_k\}}{\sum_{\vec{z}}Q(\vec{z})\sum_{t=1}^T1\{z_t=s_j\}}
\end{aligned}
$$

不幸的是，这些总和都超过了所有可能的标签$\vec{z}\in S^T$。但是回忆一下在最后一个时间步时，在有参数矩阵分别为$A,B$的情况下，$Q(\vec{z})$在E-step中被定义为$P(\vec{z}|\vec{x};A,B)$。首先，让我们来考虑如何根据向前向后概率，$\alpha_i(t)$以及$\beta_j(t)$来表达$\hat{A}_{ij}$的分子。

$$
\begin{aligned}
& \sum_{\vec{z}}Q(\vec{z})\sum_{t=1}^T1\{z_{t-1}=s_i\wedge z_t=s_j\} \\
=& \sum_{t=1}^T\sum_{\vec{z}}1\{z_{t-1}=s_i\wedge z_t=s_j\}Q(\vec{z}) \\
=& \sum_{t=1}^T\sum_{\vec{z}}1\{z_{t-1}=s_i\wedge z_t=s_j\}P(\vec{z}|\vec{x};A,B) \\
=& \frac 1{P(\vec{x};A,B)}\sum_{t=1}^T\sum_{\vec{z}}1\{z_{t-1}=s_i\wedge z_t=s_j\}P(\vec{z},\vec{x};A,B) \\
=& \frac 1{P(\vec{x};A,B)}\sum_{t=1}^T\alpha_i(t)A_{ij}B_{jx_t}\beta_j(t+1)
\end{aligned}
$$

在前两步骤中，我们重新数学符号，并在式中代入$Q$的定义，然后我们在第$4$行的推导中使用了贝叶斯规则，随后在第$5$行中代入对$\alpha,\beta,A$和$B$的定义。类似地，分母可以用分子对$j$求和来表示。

$$
\begin{aligned}
& \sum_{\vec{z}}Q(\vec{z})\sum_{t=1}^T1\{z_{t-1}=s_i\} \\
=& \sum_{j=1}^{|S|}\sum_{\vec{z}}Q(\vec{z})\sum_{t=1}^T1\{z_{t-1}=s_i\wedge z_t=s_j\} \\
=& \frac 1{P(\vec{x};A,B)}\sum_{j=1}^{|S|}\sum_{t=1}^T\alpha_i(t)A_{ij}B_{jx_t}\beta_j(t+1)
\end{aligned}
$$

结合这些表达式，我们可以充分描述我们的最大似然状态转换$\hat{A}_{ij}$，而不需要枚举所有可能的标签：

$$
\hat{A}_{ij} = \frac{\sum_{t=1}^T\alpha_i(t)A_{ij}B_{jx_t}\beta_j(t+1)}{\sum_{j=1}^{|S|}\sum_{t=1}^T\alpha_i(t)A_{ij}B_{jx_t}\beta_j(t+1)}
$$

同样，$\hat{B}_{jk}$的分子可以表示为：

$$
\begin{aligned}
& \sum_{\vec{z}}Q(\vec{z})\sum_{t=1}^T1\{z_t=s_j\wedge x_t=v_k\} \\
=& \frac 1{P(\vec{x};A,B)}\sum_{t=1}^T\sum_{\vec{z}}1\{z_t=s_j\wedge x_t=v_k\}P(\vec{z},\vec{x};A,B) \\
=& \frac 1{P(\vec{x};A,B)}\sum_{i=1}^{|S|}\sum_{t=1}^T\sum_{\vec{z}}1\{z_{t-1}=s_i\wedge z_t=s_j\wedge x_t=v_k\}P(\vec{z},\vec{x};A,B) \\
=& \frac 1{P(\vec{x};A,B)}\sum_{i=1}^{|S|}\sum_{t=1}^T1\{x_t=v_t\}\alpha_i(t)A_{ij}B_{jx_t}\beta_j(t+1)
\end{aligned}
$$

$\hat{B}_{jk}$的分母是：

$$
\begin{aligned}
& \sum_{\vec{z}}Q(\vec{z})\sum_{t=1}^T1\{z_t=s_j\} \\
=& \frac 1{P(\vec{x};A,B)}\sum_{i=1}^{|S|}\sum_{t=1}^T\sum_{\vec{z}}1\{z_{t-1}=s_i\wedge z_t=s_j\}P(\vec{z},\vec{x};A,B) \\
=& \frac 1{P(\vec{x};A,B)}\sum_{i=1}^{|S|}\sum_{t=1}^T\alpha_i(t)A_{ij}B_{jx_t}\beta_j(t+1)
\end{aligned}
$$

结合这些表达式，得到最大似然发射概率的形式为：

$$
\hat{B}_{jk}=\frac{\sum_{i=1}^{|S|}\sum_{t=1}^T1\{x_t=v_t\}\alpha_i(t)A_{ij}B_{jx_t}\beta_j(t+1)}{\sum_{i=1}^{|S|}\sum_{t=1}^T\alpha_i(t)A_{ij}B_{jx_t}\beta_j(t+1)}
$$

<hr style="height:1px;border:none;border-top:3px solid black;" />

**算法 3** HMM参数学习的前向后向算法： 

<hr style="height:1px;border:none;border-top:1px solid black;" />

初始化：设$A$和$B$为随机有效的概率矩阵，其中$A_{i0}=0,B_{0k}=0,i=1..|S|,k=1..|V|$

重复直到收敛：{

（$E$步）运行前向和后向算法进行计算$\alpha_i,\beta_i,i=1..|S|$，然后设：

$$
\gamma_t(i,j):=\alpha_i(t)A_{ij}B_{jx_t}\beta_j(t+1)
$$

（$M$步）重新估计最大似然参数为：

$$
\begin{aligned}
A_{ij} &:= \frac{\sum_{t=1}^T\gamma_t(i,j)}{\sum_{j=1}^{|S|}\sum_{t=1}^T\gamma_t(i,j)} \\
B_{jk} &:= \frac{\sum_{i=1}^{|S|}\sum_{t=1}^T1\{x_t=v_k\}\gamma_t(i,j)}{\sum_{i=1}^{|S|}\sum_{t=1}^T\gamma_t(i,j)}
\end{aligned}
$$

}

<hr style="height:1px;border:none;border-top:1px solid black;" />

算法$3$展示了用于HMMs中参数学习的前向后向算法或Baum-Welch算法的变体。在$E$步，我们并没有对于所有的$\vec{z}\in S^T$来明确的计算$Q(\vec{z})$，而是计算一个充分统计量$\gamma_t(i,j):=\alpha_i(t)A_{ij}B_{jx_t}\beta_j(t+1)$。对于所有观察序列$\vec{x}$，这个统计量正比于时间步$t$从状态$x_i$转移到状态$x_j$的概率。$A_{ij}$和$B_{jk}$导出的表达式在直观上很有吸引力。$A_{ij}$计算式是从状态$s_i$到$s_j$的期望数除以$s_i$出现的期望次数。同样，$B_{jk}$的计算式是$v_k$转移到$s_j$的期望数量除以$s_j$出现的预期数量。

与许多$EM$算法应用一样，HMMs的参数学习是一个具有许多局部极大值的非凸问题。$EM$算法将根据其初始参数收敛到最大值，因此可能需要多次迭代。此外，通常重要的是$A$和$B$表示的概率分布的平滑计算，以便没有转移或发射被分配为$0$的概率。

##### 2.5 扩展阅读

学习隐马尔可夫模型有很多很好的资源。对于NLP的应用，我推荐查看Jurafsky & Martin's写的《Speech and Language Processing》$^1$第二版或Manning & Schütze 写的《Foundations of Statistical Natural Language Processing.》。此外，Eisner写的HMM-in-a-spreadsheet[1]`注：参考资料[1]见文章最下方`是一种轻量级的交互方式，可以学习只需要电子表格应用程序的HMM。

>1 <a target='_blank' href='http://www.cs.colorado.edu/~martin/slp2.html'>http://www.cs.colorado.edu/~martin/slp2.html</a>

##### 参考资料

<blockquote id='[1]'>[1] Jason Eisner.<a target='_blank' href='https://dl.acm.org/citation.cfm?id=1118110'>An interactive spreadsheet for teaching the forward-backward algorithm.</a>In Dragomir Radev and Chris Brew, editors, Proceedings of the ACL Workshop on Effective Tools and Methodologies for Teaching NLP and CL, pages 10-18, 2002.</blockquote>