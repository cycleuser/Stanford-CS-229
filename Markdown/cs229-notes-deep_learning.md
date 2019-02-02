# CS229 课程讲义中文翻译
CS229 Lecture notes

|原作者|翻译|
|---|---|
|[Andrew Ng  吴恩达](http://www.andrewng.org/),Kian Katanforoosh|[CycleUser](https://www.zhihu.com/people/cycleuser/columns)|

|相关链接|
|---|
|[Github 地址](https://github.com/Kivy-CN/Stanford-CS-229-CN)|
|[知乎专栏](https://zhuanlan.zhihu.com/MachineLearn)|
|[斯坦福大学 CS229 课程网站](http://cs229.stanford.edu/)|
|[网易公开课中文字幕视频](http://open.163.com/movie/2008/1/M/C/M6SGF6VB4_M6SGHFBMC.html)|

# 深度学习(Deep Learning)

现在开始学深度学习.在这部分讲义中,我们要简单介绍神经网络,讨论一下向量化以及利用反向传播(backpropagation)来训练神经网络.

## 1 神经网络(Neural Networks)

我们将从小处开始逐渐构建一个神经网络,一步一步来.回忆一下最开始本课程的时候就见到的那个房价预测模型:给定房屋的面积,我们要预测其价格.

在之前的章节中,我们学到的方法是在数据图像中拟合一条直线.现在咱们不再拟合直线了,而是通过设置绝对最低价格为零来避免出现有负值房价出现.这就在图中让直线拐了个弯,如图1所示.

我们的目标是输入某些个输入特征$x$到一个函数$f(x)$中,然后输出房子$y$的价格.规范来表述就是:$f:x\rightarrow y$.可能最简单的神经网络就是定义一个单个神经元(neuron)的函数$f(x)$,使其满足$f(x)=\max(ax+b,0)$,其中的$a,b$是参数(coefficients).这个$f(x)$所做的就是返回一个单值:要么是$(ax+b)$,要么是0,就看哪个更大.在神经网络的领域,这个函数叫做一个ReLU(英文读作'ray-lu'),或者叫整流线性单元(rectified linear unit).更复杂的神经网络可能会使用上面描述的单个神经元然后堆栈(stack)起来,这样一个神经元的输出就是另一个神经元的输入,这就得到了一个更复杂的函数.

现在继续深入房价预测的例子.除了房子面积外,假如现在你还指导了卧房数目,邮政编码,以及邻居的财富状况.构建神经网络的国产和乐高积木(Lego bricks)差不多:把零散的砖块堆起来构建复杂结构而已.同样也适用于神经网络:选择独立的神经元并且对战起来创建更复杂的神经元网络

有了上面提到的这些特征(面积,卧房数,邮编,社区财富状况),就可以决定这个房子的价格是否和其所能承担的最大家庭规模有关.

## 2 向量化(Vectorization)

### 2.1 输出计算向量化(Vectorizing the Output Computation)

### 2.2 训练样本集向量化(Vectorization Over Training Examples)

## 3 反向传播(Backpropagation)

### 3.1 参数初始化(ParameterInitialization)

### 3.2 优化(Optimization)

### 3.3 参数分析(Analyzing the Parameters)

#### 3.3.1 L2规范化(Regularization)

#### 3.3.2 参数共享(Parameter Sharing)