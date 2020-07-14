# CS229 课程讲义中文翻译
CS229 Lecture notes

|原作者|翻译|校对|
|---|---|---|
|Dan Boneh ， [Andrew Ng  吴恩达](http://www。andrewng。org/)|[CycleUser](https://www。zhihu。com/people/cycleuser/columns)|[XiaoDong_Wang](https://github.com/Dongzhixiao) |


|相关链接|
|---|
|[Github 地址](https://github。com/Kivy-CN/Stanford-CS-229-CN)|
|[知乎专栏](https://zhuanlan。zhihu。com/MachineLearn)|
|[斯坦福大学 CS229 课程网站](http://cs229。stanford。edu/)|
|[网易公开课中文字幕视频](http://open。163。com/movie/2008/1/M/C/M6SGF6VB4_M6SGHFBMC。html)|


# 第十三章

### 第十四部分 线性二次调节，微分动态规划，线性二次高斯分布

上面三个名词的英文原版分别为:
1. Linear Quadratic Regulation，缩写为LQR；
2. Differential Dynamic Programming，缩写为DDP；
3. Linear Quadratic Gaussian，缩写为LQG。

#### 1 有限范围马尔科夫决策过程(Finite-horizon MDPs)

前面关于强化学习(Reinforcement Learning)的章节中，我们定义了马尔科夫决策过程(Markov Decision Processes，缩写为MDPs)，还涉及到了简单情景下的值迭代(Value Iteration)和策略迭代(Policy Iteration)。还具体介绍了**最优贝尔曼方程(optimal Bellman equation)，** 这个方程定义了对应最优策略(optimal policy)$\pi^*$的最优值函数(optimal value function)$V^{\pi^*}$。

$$
V^{\pi^*}(s)=R(s)+\max_{a \in \mathcal{A}} \gamma \sum_{s' \in \mathcal{S}} P_{sa}(s')V^{\pi^*}(s')
$$

通过优化值函数，就可以恢复最优策略$\pi^*$:

$$
\pi^*(s)=\arg\max_{a\in \mathcal{A}} \sum_{s'\in \mathcal{S}} P_{sa} (s')V^{\pi^*}(s')
$$

本章的讲义将会介绍一个更通用的情景:

1. 这次我们希望写出来的方程能够对离散和连续的案例都适用。因此就要用期望$E_{s' \sim P_{sa}}[V^{\pi^*}(s')]$替代求和$\sum_{s'\in S} P_{sa}(s')V^{\pi^*}(s')$。这就意味着在下一个状态中使用值函数的期望(exception)。对于离散的有限案例，可以将期望写成对各种状态的求和。在连续场景，可以将期望写成积分(integral)。上式中的记号$s'\sim P_{sa}$的意思是状态$s'$是从分布$P_{sa}$中取样得到的。

2. 接下来还假设奖励函数(reward)**同时依赖状态(states)和动作(actions)。** 也就是说，$R:\mathcal{S}\times\mathcal{A} \rightarrow R$。这就意味着前面计算最优动作的方法改成了

$$
\pi^*(s)=\arg\max_{a\in A} R(s，a)+\gamma E_{s'\sim P_{sa}}[V^{\pi^*}(s')]
$$

3. 以前我们考虑的是一个无限范围马尔科夫决策过程(infinite horizon MDP)，这回要改成**有限范围马尔科夫决策过程(finite horizon MDP)，** 定义为一个元组(tuple):

$$
(\mathcal{S},\mathcal{A},P_{sa},T,R)
$$

其中的$T>0$的**时间范围(time horizon)，** 例如$T=100$。这样的设定下，支付函数(payoff)就变成了:

$$
R(s_0,a_0)+R(s_1,a_1)+\dots+R(s_T,a_T)
$$

而不再是之前的:

$$
\begin{aligned}
& R(s_0,a_0)+\gamma R(s_1,a_1) + \gamma^2 R(s_2,a_2)+\dots\\
& \sum^\infty_{t=0}R(s_t,a_t)\gamma^t
\end{aligned}
$$

折扣因子(discount factor)$\gamma$哪去了呢？还记得当初引入这个$\gamma$的一部分原因就是由于要保持无穷项求和(infinite sum)是有限值(finite)并且好定义(well-defined)的么？如果奖励函数(rewards)绑定了一个常数$\bar R$，则支付函数(payoff)也被绑定成:

$$
|\sum^{\infty}_{t=0}R(s_t)\gamma^t|\le \bar R \sum^{\infty}_{t=0}\gamma^t
$$

这就能识别是一个几何求和(geometric sum)！现在由于支付函数(payoff)是一个有限和(finite sum)了，那折扣因子(discount factor)$\gamma$就没有必要再存在了。

在这种新环境下，事情就和之前不太一样了。首先是最优策略(optimal policy)$\pi^*$可能是非稳定的(non-stationary)，也就意味着它可能**随着时间步发生变化。** 也就是说现在有:

$$
\pi^{(t)}:\mathcal{S}\rightarrow\mathcal{A}
$$

上面括号中的$(t)$表示了在第$t$步时候的策略函数(policy)。遵循策略$\pi^{(t)}$的有限范围马尔科夫决策过程如下所示：开始是某个状态$s_0$，然后对应第$0$步时候的策略$\pi^{(0)}$采取某种行为$a_0:= \pi^{(0)}(s_0)$。然后马尔科夫决策过程(MDP)转换到接下来的$s_1$，根据$P_{s_0a_0}$来进行调整。然后在选择遵循第$1$步的新策略$\pi^{(1)}$的另一个行为$a_1:= \pi^{(1)}(s_1)$。依次类推进行下去。

为什么在有限范围背景下的优化策略函数碰巧就是非稳定的呢？直观来理解，由于我们只能够选择有限的应对行为，我们可能要适应不同环境的不同策略，还要考虑到剩下的时间(步骤数)。设想有一个网格，其中有两个目标，奖励值分别是$+1$和$+10$。那么开始的时候我们的行为肯定是瞄准了最高的奖励$+10$这个目标。但如果过了几步之后，我们更靠近$+1$这个目标而没有足够的剩余步数去到达$+10$这个目标，那更好的策略就是改为瞄准$+1$了。

4. 这样的观察就使得我们可以使用对**时间依赖的方法(time dependent dynamics):**

$$
s_{t+1} \sim P^{(t)}_{s_t,a_t}
$$

这就意味着变换分布(transition distribution)$P^{(t)}_{s_t,a_t}$随着时间而变化。对$R^{(t)}$而言也是如此。要注意，现在这个模型就更加符合现实世界的情况了。比如对一辆车来说，油箱会变空，交通状况会变化，等等。结合前面提到的内容，就可以使用下面这个通用方程(general formulation)来表达我们的有限范围马尔科夫决策过程(finite horizon MDP):

$$
(\mathcal{S},\mathcal{A},P^{(t)}_{sa},T,R^{(t)})
$$

**备注：** 上面的方程其实和在状态中加入时间所得到的方程等价。

在时间$t$对于一个策略$\pi$的值函数也得到了定义，也就是从状态$s$开始遵循策略$\pi$生成的轨道(trajectories)的期望(expectation)。

$$
V_t(s)=E[R^{(t)}(s_t,a_t)+\dots+R^{(T)}(s_T,a_T)|s_t=s,\pi ]
$$

现在这个方程就是：在有限范围背景下，如何找到最优值函数(optimal value function):

$$
V^*_t(s)=\max_{\pi}V^{\pi}_t(s)
$$

结果表明对值迭代(Value Iteration)的贝尔曼方程(Bellman's equation)正好适合**动态规划(Dynamic Programming)。** 这也没啥可意外的，因为贝尔曼(Bellman)本身就是动态规划的奠基人之一，而贝尔曼方程(Bellman equation)和这个领域有很强的关联性。为了理解为啥借助基于迭代的方法(iteration-based approach)就能简化问题，我们需要进行下面的观察:

1. 在游戏终结（到达步骤$T$）的时候，最优值(optimal value)很明显就是

$$
\forall s\in \mathcal{S}: V^*_T(s):=\max_{a\in A} R^{(T)}(s,a) \qquad(1)
$$

2. 对于另一个时间步$0\le t <T$，如果假设已经知道了下一步的最优值函数$V^*_{t+1}$，就有:

$$
\forall t<T，s \in \mathcal{S}: V^*_t (s):= \max_{a\in A} [R^{(t)}(s,a)+E_{s'\sim P^{(t)}_{sa}}[V^*_{t+1}(s')]] \qquad (2)
$$

观察并思考后，就能想出一个聪明的算法来解最优值函数了:

1. 利用等式$(1)$计算$V^*_T$。
2. for $t= T-1,\dots,0$:
&emsp;&emsp;使用$V^*_{t+1}$利用等式$(2)$计算$V^*_t$。

**备注:** 可以将标准值迭代(standard value iteration)看作是上述通用情况的一个特例，就是不用记录时间(步数)。结果表明在标准背景下，如果对$T$步骤运行值迭代，会得到最优值迭代的一个$\gamma^T$的近似(几何收敛，geometric convergence)。参考习题集4中有对下列结果的证明:

<u>定理：</u>设$B$表示贝尔曼更新函数(Bellman update)，以及$||f(x)||_\infty:= \sup_x|f(x)|$。如果$V_t$表示在第$t$步的值函数，则有:

$$
\begin{aligned}
||V_{t+1}-V^*||_\infty &=||B(V_t)-V^*||_\infty\\
&\le \gamma||V_t-V^*||_\infty\\
&\le \gamma^t||V_1-V^*||_\infty
\end{aligned}
$$

也就是说贝尔曼运算器$B$成了一个$\gamma$收缩算子(contracting operator)。

#### 2 线性二次调节(Linear Quadratic Regulation，缩写为LQR)

在本节，我们要讲一个上一节所提到的有限范围(finite-horizon)背景下**精确解(exact solution)** 很容易处理的特例。这个模型在机器人领域用的特别多，也是在很多问题中将方程化简到这一框架的常用方法。

首先描述一下模型假设。考虑一个连续背景，都用实数集了:

$$
\mathcal{S}=R^n,\quad\mathcal{A}=R^d
$$

然后设有噪音(noise)的**线性转换(linear transitions):**

$$
s_{t+1}=A_ts_t+B_ta_t+w_t
$$

上式中的$A_t\in R^{n\times n}，B_t\in R^{n\times d}$实矩阵，而$w_t\sim N(0，\Sigma_t)$是某个高斯分布的噪音（均值为**零**）。我们接下来要讲的内容就表明：只要噪音的均值是$0$，就不会影响最优化策略。

另外还要假设一个**二次奖励函数(quadratic rewards):**

$$
R^{(t)}(s_t,a_t)=-s_t^TU_ts_t-a_t^TW_ta_t
$$

上式中的$U_t\in R^{n\times n}，W_t\in R^{n\times d}$都是正定矩阵(positive definite matrices)，这就意味着奖励函数总是**负的(negative)。**

**要注意**这里的奖励函数的二次方程(quadratic formulation)就等价于无论奖励函数是否更高我们都希望能接近原始值(origin)。例如，如果$U_t=I_n$就是$n$阶单位矩阵(identity matrix)，而$W_t=I_d$为一个$d$阶单位矩阵，那么就有$R_t=-||s_t||^2-||a_t||^2$，也就意味着我们要采取光滑行为(smooth actions)（$a_t$的范数(norm)要小）来回溯到原始状态（$s_t$的范数(norm)要小）。这可以模拟一辆车保持在车道中间不发生突发运动。

接下来就可以定义这个线性二次调节(LQR)模型的假设了，这个LQR算法包含两步骤:

**第一步**设矩阵$A，B，\Sigma$都是未知的。那就得估计他们，可以利用强化学习课件中的值估计(Value Approximation)部分的思路。首先是从一个任意策略(policy)收集转换(collect transitions)。然后利用线性回归找到$\arg\min_{A,B}\sum^m_{i=1}\sum^{T-1}_{t=0}||s^{(i_)}_{t+1}- ( As^{(i)}_t +Ba^{(i)}_t)||^2$。最后利用高斯判别分析(Gaussian Discriminant Analysis，缩写为GDA)中的方法来学习$\Sigma$。

**第二步**假如模型参数已知了，比如可能是给出了，或者用上面第一步估计出来了，就可以使用动态规划(dynamic programming)来推导最优策略(optimal policy)了。

也就是说，给出了:

$$
\begin{cases}
s_{t+1} &= A_ts_t+B_ta_t+w_t\qquad 已知A_t,B_t,U_t,W_t,\Sigma_t\\
R^{(t)}(s_t，a_t)&= -s_t^TU_ts_t-a^T_tW_ta_t
\end{cases}
$$

然后要计算出$V_t^*$。如果回到第一步，就可以利用动态规划，就得到了:

1. **初始步骤(Initialization step)**

&emsp;&emsp;对最后一次步骤$T$，

$$
\begin{aligned}
V^*_T(s_T)&=\max_{a_T\in A}R_T(s_T,a_T)\\
&=\max_{a_T\in A}-s^T_TU_ts_T - a^T_TW_ta_T\\
&= -s^T_TU_ts_T\qquad\qquad(\text{对}a_T=0\text{最大化})
\end{aligned}
$$

&emsp;&emsp; **(译者注:原文这里第一步的第二行公式中用的是$U_T$，应该是写错了，结合上下文公式推导来看，分明应该是$U_t$)**

2. **递归步骤(Recurrence step)**

&emsp;&emsp;设$t<T$。加入已经知道了$V^*_{t+1}$。

<u>定理1:</u>很明显如果$V^*_{t+1}$是$s_t$的一个二次函数，则$V_t^*$也应该是$s_t$的一个二次函数。也就是说，存在某个矩阵$\Phi$以及某个标量$\Psi$满足:

$$
\begin{aligned}
\text{if} \quad V^*_{t+1}(s_{t+1}) &= s^T_{t+1}\Phi_{t+1}s_{t+1}+\Psi_{t+1}\\
\text{then} \quad V^*_t(s_t)&=s^T_t\Phi_ts_t+\Psi_t
\end{aligned}
$$

对时间步骤$t=T$，则有$\Phi_t=-U_T，\Psi_T=0$。

<u>定理2:</u>可以证明最优策略是状态的一个线性函数。

已知$V^*_{t+1}$就等价于知道了$\Phi_{t+1}，\Psi_{t+1}$，所以就只需要解释如何从$\Phi_{t+1}，\Psi_{t+1}$去计算$\Phi_{t}，\Psi_{t}$，以及问题中的其他参数。

$$
\begin{aligned}
V^*_t(s_t)&=  s_t^T\Phi_ts_t+\Psi_t \\
&= \max_{a_t}[R^{(t)}(s_t,a_t)+E_{s_{t+1}\sim P^{(t)}_{s_t,a_t}}[V^*_{t+1}(s_{t+1})]]  \\
&= \max_{a_t}[-s_t^TU_ts_t-a_t^TV_ta_t+E_{s_{t+1}\sim N(A_ts_t+B_ta_t,\Sigma_t)}  [s_{t+1}^T\Phi_{t+1}s_{t+1}+\Psi_{t+1}] ]  \\
\end{aligned}
$$

上式中的第二行正好就是最优值函数(optimal value function)的定义，而第三行是通过代入二次假设和模型方法。注意最后一个表达式是一个关于$a_t$的二次函数，因此很容易就能优化掉$^1$。然后就能得到最优行为(optimal action)$a^*_t$:

>1 这里用到了恒等式(identity)$E[w_t^T\Phi_{t+1}w_t] =Tr(\Sigma_t\Phi_{t+1})，\quad \text{其中} w_t\sim N(0，\Sigma_t)$)。

$$
\begin{aligned}
a^*_t&= [(B_t^T\Phi_{t+1}B_t-V_t)^{-1}B_t\Phi_{t+1}A_t]\cdot s_t\\
&= L_t\cdot s_t\\
\end{aligned}
$$

上式中的
$$
L_t := [(B_t^T\Phi_{t+1}B_t-V_t)^{-1}B_t\Phi_{t+1}A_t]
$$

这是一个很值得注意的结果(impressive result)：优化策略(optimal policy)是关于状态$s_t$的**线性函数。** 对于给定的$a_t^*$，我们就可以解出来$\Phi_t$和$\Psi_t$。最终就得到了**离散里卡蒂方程(Discrete Ricatti equations):**

$$
\begin{aligned}
\Phi_t&= A^T_t(\Phi_{t+1}-\Phi_{t+1}B_t(B^T_t\Phi_{t+1}B_t-W_t)^{-1}B_t\Phi_{t+1})A_t-U_t\\
\Psi_t&= -tr(\Sigma_t\Phi_{t+1})+\Psi_{t+1}\\
\end{aligned}
$$

<u>定理3:</u>要注意$\Phi_t$既不依赖$\Psi_t$也不依赖噪音项$\Sigma_t$！由于$L_t$是一个关于$A_t，B_t，\Phi_{t+1}$的函数，这就暗示了最优策略也**不依赖噪音！** （但$\Psi_t$是依赖$\Sigma_t$的，这就暗示了最优值函数$V^*_t$也是依赖噪音$\Sigma_t$的。）

然后总结一下，线性二次调节(LQR)算法就如下所示:

1. 首先，如果必要的话，估计参数$A_t,B_t,\Sigma_t$。
2. 初始化$\Phi_T:=-U_T,\quad \Psi_T:=0$。
3. 从$t=T-1,\dots,0$开始迭代，借助离散里卡蒂方程(Discrete Ricatti equations)来利用$\Phi_{t+1},\Psi_{t+1}$来更新$\Phi_{t},\Psi_{t}$，如果存在一个策略能朝着$0$方向推导状态，收敛就能得到保证。

利用<u>定理3</u>，我们知道最优策略不依赖与$\Psi_t$而只依赖$\Phi_t$，这样我们就可以**只** 更新$\Phi_t$，从而让算法运行得更快一点！

#### 3 从非线性方法(non-linear dynamics)到线性二次调节(LQR)

很多问题都可以化简成线性二次调节(LDR)的形式，包括非线性的模型。LQR是一个很好的方程，因为我们能够得到很好的精确解，但距离通用还有一段距离。我们以倒立摆(inverted pendulum)为例。状态的变换如下所示:

$$
\begin{pmatrix}
x_{t+1}\\
\dot x_{t+1}\\
\theta_{t+1}\\
\dot \theta_{t+1}
\end{pmatrix}=F\begin{pmatrix} \begin{pmatrix} x_t\\
 \dot x_t\\
  \theta_t\\
   \dot\theta_t \end{pmatrix}，a_t\end{pmatrix}
$$

其中的函数$F$依赖于角度余弦等等。然后这个问题就成了：

$$
我们能将这个系统线性化么?
$$

##### 3.1 模型的线性化(Linearization of dynamics)

假设某个时间$t$上，系统的绝大部分时间都处在某种状态$\bar s_t$上，而我们要选取的行为大概就在$\bar a_t$附近。对于倒立摆问题，如果我们达到了某种最优状态，就会满足：行为很小并且和竖直方向的偏差不大。

这就要用到泰勒展开(Taylor expansion)来将模型线性化。简单的情况下状态是一维的，这时候转换函数$F$就不依赖于行为，这时候就可以写成:

$$
s_{t+1}=F(s_t)\approx F(\bar s_t)+ F'(\bar s_t)\cdot (s_t-\bar s_t)
$$

对于更通用的情景，方程看着是差不多的，只是用梯度(gradients)替代简单的导数(derivatives):

$$
s_{t+1}\approx F(\bar s_t,\bar a_t)+\nabla _sF(\bar s_t,\bar a_t)\cdot (s_t-\bar s_t)+\nabla_aF(\bar s_t,\bar a_t)\cdot (a_t-\bar a_t) \qquad \text{(3)}
$$

现在$s_{t+1}$就是关于$s_t，a_t$的线性函数了，因为可以将等式$(3)$改写成下面的形式：

$$
s_{t+1}\approx As_t+Ba_t+k
$$

**(译者注:原文这里的公式应该是写错了，写成了$s_{t+1}\approx As_t+Bs_t+k$)**

上式中的$k$是某个常数，而$A，B$都是矩阵。现在这个写法就和在LQR里面的假设非常相似了。这时候只要摆脱掉常数项$k$就可以了!结果表明只要任意增长一个维度就可以将常数项吸收进$s_t$中区。这和我们在线性回归的课程里面用到的办法一样。

##### 3.2 微分动态规划(Differential Dynamic Programming，缩写为DDP)

如果我们的目标就是保持在某个状态$s^*$，上面的方法都能够很适合所选情景（比如倒立摆或者一辆车保持在车道中间）。不过有时候我们的目标可能要更复杂很多。

本节要讲的方法适用于要符合某些轨道的系统（比如火箭发射）。这个方法将轨道离散化称为若干离散的时间步骤，然后运用前面的方法创建中间目标!这个方法就叫做**微分动态规划(Differential Dynamic Programming，缩写为DDP)。** 主要步骤包括：

**第一步**利用简单控制器(naive controller)创建一个标称轨道(nominal trajectory)，对要遵循轨道进行近似。也就是说，我们的控制器可以用如下方法来近似最佳轨道：

$$
s^*_0,a^*_0\rightarrow s^*_1,a^*_1\rightarrow\dots
$$

**第二步**在每个轨道点(trajectory point)$s^*_t$将模型线性化，也就是:

$$
s_{t+1}\approx F(s^*_t,a^*_t)+\nabla_s F(s^*_t,a^*_t)(s_t-s^*_t)+\nabla_aF(s^*_t,a^*_t)(a_t-a^*_t)
$$

上面的$s_t,a_t$是当前的状态和行为。现在已经在每个轨道点都有了线性估计了，就可以使用前面的方法将其改写成:

$$
s_{t+1}=A_t\cdot s_t+B_t\cdot a_t
$$

（要注意在这个情况下，我们可以使用在本章一开头所提到的非稳定动力学模型背景。）

**注意，** 这里我们可以对奖励函数(reward)$R^{(t)}$推导一个类似的积分(derivation)，使用一个二阶泰勒展开(second-order Taylor expansion)就可以了。

$$
\begin{aligned}
R(s_t，a_t)& \approx R(s^*_t，a^*_t)+\nabla_s R(s^*_t，a^*_t)(s_t-s^*_t) +\nabla_a R(s^*_t，a^*_t)(a_t-a^*_t) \\
& + \frac{1}{2}(s_t-s^*_t)^TH_{ss}(s_t-s^*_t)+(s_t-s^*_t)^TH_{sa}(a_t-a^*_t)\\
&  + \frac{1}{2}(a_t-a^*_t)^TH_{aa}(a_t-a^*_t) \\
\end{aligned}
$$

上式中的$H_{xy}$表示的 $R$ 的海森矩阵(Hessian)项，对应的$x$和$y$是在$(s^*_t,a^*_t)$中得到的（略去不表）。这个表达式可以重写成:

$$
R_t(s_t，a_t)= -s_t^TU_ts_t-a_t^TW_ta_t
$$

对于某些矩阵$U_t,W_t$，可以再次使用扩展维度的方法。注意:

$$
\begin{pmatrix} 1&x \end{pmatrix}\cdot \begin{pmatrix} a& b\\c&d \end{pmatrix} \cdot \begin{pmatrix} 1\\x \end{pmatrix} = a+2bx+cx^2
$$

**第三步**现在你就能够相信这个问题可以**严格**写成LQR框架的形式了吧。然后就可以利用线性二次调节(LQR)来找到最优策略$\pi_t$。这样新的控制器就会更好些！

**注意:** 如果LQR轨道和线性近似的轨道偏离太远，可能会出现一些问题，不过这些都可以通过调节奖励函数形态来进行修正...

**第四步**现在就得到了一个新的控制器了（新的策略$\pi_t$），使用这个新控制器来产生一个新的轨道:

$$
s^*_0,\pi_0(s^*_0)\rightarrow s^*_1,\pi_1(s^*_1)\rightarrow \quad \rightarrow s^*_T
$$

注意当我们生成了这个新的轨道的时候，使用真实的$F$而不是其线性估计来计算变换，这就意味着:

$$
s^*_{t+1}=F(s^*_t，a^*_t)
$$

然后回到第二步，重复，直到达到某个停止条件(stopping criterion)。

#### 4 线性二次高斯分布(Linear Quadratic Gaussian，缩写为LQG)

在现实是集中我们可能没办法观测到全部的状态$s_t$。例如一个自动驾驶的汽车只能够从一个相机获取一个图像，这就是一次观察了，而不是整个世界的全部状态。目前为止都是假设所有状态都可用。可是在现实世界的问题中并不见得总是如此，我们需要一个新工具来对这种情况进行建模：部分观测的马尔科夫决策过程(Partially Observable MDPs，缩写为POMDP)。

POMDP是一个带有了额外观察层的马尔科夫决策过程(MDP)。也就是说要加入一个新变量$o_t$，在给定的当前状态下这个$o_t$遵循某种条件分布:

$$
o_t|s_t\sim O(o|s)
$$

最终，一个有限范围的部分观测的马尔科夫决策过程(finite-horizon POMDP)就是如下所示的一个元组(tuple):

$$
(\mathcal{S},\mathcal{O},\mathcal{A},P_{sa},T,R)
$$

在这个框架下，整体的策略就是要在观测$o_1,o_2,\dots,o_t$的基础上，保持一个**置信状态（belief state，对状态的分布）。** 这样在PDMDP中的策略就是从置信状态到行为的映射。

在本节，我们队LQR进行扩展以适应新的环境。假设我们观测的是$y_t\in R^m$，其中的$m<n$，且有:

$$
\begin{cases}
y_t &= C\cdot s_t +v_t\\
s_{t+1} &=  A\cdot s_t+B\cdot a_t+ w_t\\
\end{cases}
$$

上式中的$C\in R^{m\times n}$是一个压缩矩阵(compression matrix)，而$v_t$是传感器噪音（和$w_t$类似也是高斯分布的）。要注意这里的奖励函数$R^{(t)}$是未做更改的，是关于状态（而不是观察）和行为的函数。另外，由于分布都是高斯分布，置信状态就也将是高斯分布的。在这样的新框架下，看看找最优策略的方法:

**第一步**首先计算可能状态（置信状态）的分布，以已有观察为基础。也就是说要计算下列分布的均值$s_{t|t}$以及协方差$\Sigma_{t|t}$:
$$
s_t|y_1 ,\dots, y_t \sim \mathcal{N}(s_{t|t},\Sigma_{t|t})
$$

为了进行时间效率高的计算，这里要用到卡尔曼滤波器算法(Kalman Filter algorithm)（阿波罗登月舱上就用了这个算法）。

**第二步**然后就有了分布了，接下来就用均值$s_{t|t}$来作为对$s_t$的最佳近似。

**第三步**然后设置行为$a_t:= L_ts_{t|t}$，其中的$L_t$来自正规线性二次调节算法(regular LQR algorithm)。

从直觉来理解，这样做为啥能管用呢？要注意到$s_{t|t}$是$s_t$的有噪音近似（等价于在LQR的基础上增加更多噪音），但我们已经证明过了LQR是独立于噪音的!

第一步就需要解释一下。这里会给出一个简单情境，其中在我们的方法里没有行为依赖性（但整体上这个案例遵循相同的思想）。设有:

$$
\begin{cases}
s_{t+1}  &= A\cdot s_t+w_t,\quad w_t\sim N(0,\Sigma_s)\\
y_t  &= C\cdot s_t+v_t,\quad v_t\sim N(0,\Sigma_y)\\
\end{cases}
$$

由于噪音是高斯分布的，可以很明显证明联合分布也是高斯分布:

$$
\begin{pmatrix}
s_1\\
\vdots\\
s_t\\
y_1\\
\vdots\\
y_t
\end{pmatrix} \sim \mathcal{N}(\mu，\Sigma) \quad\text{for some } \mu,\Sigma
$$

然后利用高斯分布的边缘方程(参考因子分析(Factor Analysis)部分的讲义)，就得到了:

$$
s_t|y_1,\dots，y_t\sim \mathcal{N}(s_{t|t},\Sigma_{t|t})
$$

可是这里使用这些方程计算边缘分布的参数需要很大的算力开销!因为这需要对规模为$t\times t$的矩阵进行运算。还记得对一个矩阵求逆需要的运算时$O(t^3)$吧，这要是在时间步骤数目上进行重复，就需要$O(t^4)$的算力开销!

**卡尔曼滤波器算法(Kalman filter algorithm)** 提供了计算均值和方差的更好的方法，只用在时间$t$上以一个**固定的时间(constant time)** 来更新！卡尔曼滤波器算法有两个基础步骤。加入我们知道了分布$s_t|y_1,\dots,y_t$:

$$
\begin{aligned}
\text{预测步骤(predict step) 计算} & s_{t+1}|y_1,\dots,y_t
\\
\text{更新步骤(update step) 计算} & s_{t+1}|y_1,\dots,y_{t+1}
\end{aligned}
$$

然后在时间步骤上迭代！预测和更新这两个步骤的结合就更新了我们的置信状态，也就是说整个过程大概类似:

$$
(s_{t}|y_1,\dots,y_t)\xrightarrow{predict} (s_{t+1}|y_1,\dots,y_t)
 \xrightarrow{update} (s_{t+1}|y_1,\dots,y_{t+1})\xrightarrow{predict}\dots
$$

**预测步骤** 假如我们已知分布:

$$
s_{t}|y_1,\dots,y_t\sim \mathcal{N}(s_{t|t},\Sigma_{t|t})
$$

然后在下一个状态上的分布也是一个高斯分布:

$$
s_{t+1}|y_1,\dots,y_t\sim \mathcal{N}(s_{t+1|t},\Sigma_{t+1|t})
$$

其中有:

$$
\begin{cases}
s_{t+1|t}&=  A\cdot s_{t|t}\\
\Sigma_{t+1|t} &= A\cdot \Sigma_{t|t}\cdot A^T+\Sigma_s
\end{cases}
$$

**更新步骤** 给定了$s_{t+1|t}$和$\Sigma_{t+1|t}$，则有:

$$
s_{t+1}|y_1,\dots,y_t\sim \mathcal{N}(s_{t+1|t},\Sigma_{t+1|t})
$$
可以证明有:

$$
s_{t+1}|y_1,\dots,y_{t+1}\sim \mathcal{N}(s_{t+1|t+1},\Sigma_{t+1|t+1})
$$

其中有:

$$
\begin{cases}
s_{t+1|t+1}&= s_{t+1|t}+K_t(y_{t+1}-Cs_{t+1|t})\\
\Sigma_{t+1|t+1} &=\Sigma_{t+1|t}-K_t\cdot C\cdot \Sigma_{t+1|t}
\end{cases}
$$

上式中的

$$
K_t:= \Sigma_{t+1|t} C^T (C \Sigma_{t+1|t} C^T + \Sigma_y)^{-1}
$$

这个矩阵$K_t$就叫做**卡尔曼增益(Kalman gain)。**

现在如果我们仔细看看方程就会发现根本不需要对时间步骤 $t$ 有观测先验。更新步骤只依赖与前面的分布。综合到一起，这个算法最开始向前运行传递计算$K_t,\Sigma_{t|t},s_{t|t}$（有时候在文献中被称为$\hat s$）。然后就向后运行（进行LQR更新）来计算变量$\Phi_t,\Psi_t,L_t$了，最终就得到了最优策略$a^*_t=L_Ts_{t|t}$。
