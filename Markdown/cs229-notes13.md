# CS229 课程讲义中文翻译
CS229 Lecture notes

|原作者|翻译|
|---|---|
|Dan Boneh , [Andrew Ng  吴恩达](http://www.andrewng.org/)|[CycleUser](https://www.zhihu.com/people/cycleuser/columns)|


|相关链接|
|---|
|[Github 地址](https://github.com/Kivy-CN/Stanford-CS-229-CN)|
|[知乎专栏](https://zhuanlan.zhihu.com/MachineLearn)|
|[斯坦福大学 CS229 课程网站](http://cs229.stanford.edu/)|
|[网易公开课中文字幕视频](http://open.163.com/movie/2008/1/M/C/M6SGF6VB4_M6SGHFBMC.html)|


# 第十三章

## Part XIV（第十四部分?）

## 线性二次调节,微分动态规划,线性二次高斯分布
上面三个名词的英文原版分别为:
1. Linear Quadratic Regulation,缩写为LQR;
2. Differential Dynamic Programming,缩写为DDP;
3. 3Linear Quadratic Gaussian,缩写为LQG.

## 1 有限范围马尔科夫决策过程(Finite-horizon MDPs)

前面关于强化学习(Reinforcement Learning)的章节中,我们定义了马尔科夫决策过程(Markov Decision Processes,缩写为MDPs),还涉及到了简单情景下的值迭代(Value Iteration)和策略迭代(Policy Iteration).还具体介绍了最优贝尔曼方程(optimal Bellman equation),这个方程定义了对应最优策略(optimal policy)$\pi*$的最优值函数(optimal value function)$V^{\pi^*}$.

$$
V^{\pi^*}(s)=R(s)+\max_{a \in A} \gamma \sum_{s' \in S} P_{sa}(s')V^{\pi^*}(s')
$$

通过优化值函数,就可以恢复最优策略$\pi^*$:

$$
\pi^*(s)=\arg\max_{a\in A} \sum_{s'\in S} P_{sa} (s')V^{\pi^*}(s')
$$

本章的讲义将会介绍一个更通用的情景:

1. 这次我们希望写出来的方程能够对离散和连续的案例都适用.因此就要用期望$E_{s' \sim P_{sa}}[V^{\pi^*}(s')]$替代求和$\sum_{s'\in S} P_{sa}(s')V^{\pi^*}(s')$.这就意味着在下一步中使用值函数的期望(exception).对于离散的有限案例,可以将期望写成对各种状态的求和.在连续场景,可以将期望写成积分(integral).上式中的记号$s'\sim P_{sa}$的意思是状态$s'$是从分布$P_{sa}$中取样得到的.

2. 接下来还假设奖励函数(reward)同时依赖状态(states)和行为(actions).也就是说,$R:S\times A \rightarrow R$.这就意味着前面计算最优行为的方法改成了$\pi^*(s)=\arg\max_{a\in A} R(s,a)+\gamma E_{s'\sim P_{sa}}[V^{\pi^*}(s')]$.

3. 以前我们考虑的是一个无限范围马尔科夫决策过程(infinite horizon MDP),这回要改成有限范围马尔科夫决策过程(finite horizon MDP),定义为一个元组(tuple):

    $$
    (S,A,P_{sa},T,R)
    $$

    其中的$T>0$的时间范围(time horizon),例如$T=100$.这样的设定下,支付函数(payoff)就变成了:

    $$
    R(s_0,a_0)+R(s_1,a_1)+...+R(s_T,a_T)
    $$

    而不再是之前的:

    $$
    \begin{aligned}
    & R(s_0,a_0)+\gamma R(s_1,a_1)++ \gamma^2 R(s_T,a_T)+...\\
    & \sum^\infty_{t=0}R(s_t,a_t)\gamma^t
    \end{aligned}
    $$

    折扣因子(discount factor)$\gamma$哪去了呢?这是要记得,当初引入这个$\gamma$的一部分原因就是由于要保持无穷项求和(infinite sum)是有限值(finite)并且好定义(well-defined).如果奖励函数(rewards)绑定了一个常数$\bar R$,则支付函数(payoff)也被绑定成:

    $$
    |\sum^{\infty}_{t==0}R(s_t)\gamma^t|\le \bar R \sum^{\infty}_{t==0}\gamma^t
    $$

    这就能识别是一个几何求和(geometric sum)!现在由于支付函数(payoff)是一个有限和(finite sum)了,那折扣因子(discount factor)$\gamma$就没有必要再存在了.

    在这种新环境下,事情就和之前不太一样了.首先是最优策略(optimal policy)$\pi^*$可能是非稳定的(non-stationary),也就意味着它可能随着时间(次数?)发生变化.也就是说现在有:

    $$
    \pi^{(t)}:S\rightarrow A
    $$

    上面括号中的$(t)$表示了在第$t$步时候的策略函数(policy).遵循策略$\pi^{(t)}$的有限范围马尔科夫决策过程如下所示:开始是某个状态$s_0$,然后对应第0步时候的策略$\pi^{(0)}$采取某种行为$a_0:= \pi^{(0)}(s_0)$.然后马尔科夫决策过程(MDP)转换到接下来的$s_1$,根据$P_{s_0a_0}$来进行调整.然后在选择遵循第1步的新策略$\pi^{(1)}$的另一个行为$a_1:= \pi^{(1)}(s_1)$.依次类推进行下去.

    为什么在有限范围背景下的优化策略函数碰巧就是非稳定的呢?直观来理解,由于我们只能够选择有限的应对行为,我们可能要适应不同环境的不同策略,还要考虑到剩下的时间(步骤数).设想有一个网格,其中有两个目标,奖励值分别是+1和+10.那么开始的时候我们的行为肯定是瞄准了最高的奖励+10这个目标.但如果过了几步之后,我们更靠近+1这个目标而没有足够的剩余步数去到达+10这个目标,那更好的策略就是改为瞄准+1了.

4. 这样的观察就使得我们可以使用对时间依赖的方法(time dependent dynamics):

    $$
    s_{t+1} \sim P^{(t)}_{s_t,a_t}
    $$

    这就意味着变换分布(transition distribution)$P^{(t)}_{s_t,a_t}$随着时间而变化.对$R^{(t)}$而言也是如此.要注意,现在这个模型就更加符合现实世界的情况了.比如对一辆车来说,油箱会变空,交通状况会变化,等等.结合前面提到的内容,就可以使用下面这个通用方程(general formulation)来表达我们的有限范围马尔科夫决策过程(finite horizon MDP):

    $$
    (S,A,P^{(t)}_{sa},T,R^{(t)})
    $$

    备注:上面的方程其实和在状态中加入时间所得到的方程等价.

    在时间$t$对于一个策略$\pi$的值函数也得到了定义,也就是从状态$s$开始遵循策略$\pi$生成的轨道(trajectories)的期望(expectation).

    $$
    V_t(s)=E[R^{(t)}(s_t,a_t)+...+R^{(T)}(s_T,a_T)|s_t=s,\pi ]
    $$

    现在这个方程就是:在有限范围背景下,如何找到最优值函数(optimal value function):

    $$
    V^*_t(s)=\max_{\pi}V^{\pi}_t(s)
    $$

结果表明对值迭代(Value Iteration)的贝尔曼方程(Bellman's equation)正好适合动态规划(Dynamic Programming).这也没啥可意外的,因为贝尔曼(Bellman)本身就是动态规划的奠基人之一,而贝尔曼方程(Bellman equation)和这个领域有很强的关联性.为了理解为啥借助基于迭代的方法(iteration-based approach)就能简化问题,我们需要进行下面的观察:

1. 在游戏终结(到达步骤T)的时候,最优值(optimal value)很明显就是
   $$
   \forall s\ in S: V^*_T(s):=\max_{a\in A} R^{(T)}(s,a) \text{\qquad (1)}
   $$
2. 对于另一个步骤t,$0\le t <T$,如果假设已经知道了下一步的最优值函数$V^*_{t+1}$,就有:
   $$
   \forall t<T,s \in S: V^*_t (s):= \max_{a\in A} [R^{(t)}(s,a)+E_{s'\sim P^{(t)}_{sa}}[V^*_{t+1}(s')]] \text{\qquad (2)}
   $$

在思想中进行了上面的观察后,就能想出一个聪明的算法来解最优值函数了:

1. 利用等式(1)计算$V^*_T$.
2. 对于 $t= T-1,...,0$:
   使用$V^*_{t+1}利用等式(2)计算$V^*_t$.
  
备注:可以将标准值迭代(standard value iteration)看作是上述通用情况的一个特例,就是不用记录时间(步数).结果表明在标准背景下,如果对T步骤运行值迭代,会得到最优值迭代的一个$\gamma^T$的近似(几何收敛,geometric convergence).参考习题集4中有对下列结果的证明:

#### 定理

设$B$表示贝尔曼更新函数(Bellman update),以及$||f(x)||_\infty:= \sup_x|f(x)|$.如果$V_t$表示在第t步的值函数,则有:

$$
\begin{aligned}
||V_{t+1}-V^*||_\infty &=||B(V_t)-V^*||_\infty\\
&\le \gamma||V_t-V^*||_\infty\\
&\le \gamma^t||V_1-V^*||_\infty
\end{aligned}
$$

也就是说贝尔曼运算器$B$成了一个$\gamma$收缩算子(contracting operator).


## 2 线性二次调节(Linear Quadratic Regulation,缩写为LQR)
