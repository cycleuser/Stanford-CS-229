# CS229 课程讲义中文翻译
CS229 Lecture notes

|原作者|翻译|校对|
|---|---|---|
| [Andrew Ng  吴恩达](http://www.andrewng.org/),Kian Katanforoosh |[CycleUser](https://www.zhihu.com/people/cycleuser/columns)| [XiaoDong_Wang](https://github.com/Dongzhixiao) |


|相关链接|
|---|
|[Github 地址](https://github.com/Kivy-CN/Stanford-CS-229-CN)|
|[知乎专栏](https://zhuanlan.zhihu.com/MachineLearn)|
|[斯坦福大学 CS229 课程网站](http://cs229.stanford.edu/)|
|[网易公开课中文字幕视频](http://open.163.com/movie/2008/1/M/C/M6SGF6VB4_M6SGHFBMC.html)|


# 关于反向传播(Backpropagation)的附加讲义

## 1 正向传播(Forward propagation)

回忆一下，给出一个输入特征$x$的时候，我们定义了$a^{[0]}=x$。然后对于层(layer)$l=1,2,3,\dots,N$，其中的$N$是网络中的层数，则有：

1. $z^{[l]}=W^{[l]}a^{[l-1]}+b^{[l]}$
2. $a^{[l]}=g^{[l]}(z^{[l]})$

在讲义中都是假设了非线性特征$g^{[l]}$对除了第$N$层以外的所有层都相同。这是因为我们可能要在输出层进行回归(regression) [因此就可能使用$g(x)=x$]，或者进行二值化分类(binary classification) [这时候用$g(x)=sigmoid(x)$]，或者是多类分类(multiclass classification) [这就要用$g(x)=softmax(x)$]。因此要将$g^{[N]}$和$g$相区分开，然后假设$g$使用在除了第$N$层外的其他所有层。

最终，给网络$a^{[N]}$的输出，为了简单记作$\hat y$，就可以衡量损失函数$J(W,b)=\mathcal{L}(a^{[N]},y)=\mathcal{L}(\hat y,y)$。例如，对于实数值的回归可以使用下面的平方损失函数(squared loss)：

$$
\mathcal{L}(\hat y,y)=\frac{1}{2} (\hat y-y)^2
$$

对使用逻辑回归(logistic regression)的二值化分类(binary classification)，我们可以使用下面的损失函数：

$$
\mathcal{L}(\hat y,y) =-[y\log \hat y+(1-y)\log(1-\hat y)]
$$

或者也可以使用负对数似然函数(negative log-likelihood)。最后是对于有$k$类的柔性最大回归(softmax regression)，使用交叉熵损失函数(cross entropy loss)：

$$
\mathcal{L}(\hat y,y)=-\sum^k_{j=1}1\{y=j\}\log\hat y_j
$$

上面这个其实就是将负对数似然函数简单扩展到多类情景而已。要注意这时候的$\hat y$是一个$k$维度向量。如果使用$y$来表示这个$k$维度向量，其中除了第$l$个位置是$1$，其他地方都是$0$。也就是为真的分类标签(label)就是$l$，这时候也可以将交叉熵损失函数以如下方式表达：

$$
\mathcal{L}(\hat y,y)=- \sum^k_{j=1}y_j\log \hat y_j
$$

## 2 反向传播(Backpropagation)

然后咱们再定义更多的记号用于反向传播。$^1$首先定义:

>1 这部分的讲义内容基于 [斯坦福大学的监督学习中关于多层神经网络算法的内容](http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/)
Scribe: Ziang Xie

$$
\delta ^{[l]} =\nabla _{z^{[l]}}\mathcal{L}(\hat y,y)
$$

然后可以按照后续内容所示来定义一个计算每个$W^{[l]},b^{[l]}$所对应的梯度的三个步骤：

1. 对于输出层$N$有:
   
   $$
   \delta ^{[N]} =\nabla _{z^{[N]}}\mathcal{L}(\hat y,y)
   $$

   有时候我们可能要直接计算$\nabla _{z^{[N]}}L(\hat y,y)$（比如$g^{[N]}$是柔性最大函数(softmax function)），而其他时候（比如$g^{[N]}$是S型函数(sigmoid function)$\sigma$）则可以应用链式法则(chain rule)：

   $$
   \nabla _{z^{[N]}}L(\hat y,y)= \nabla _{\hat y}L(\hat y,y) \circ (g^{[N]})'(z^{[N]})
   $$

   注意$(g^{[N]})'(z^{[N]})$表示的是关于$z^{[N]}$的按元素的导数(elementwise derivative)。

2. 对$l=N-1,N-1,\dots,1$则有：
   
   $$
   \delta^{[l]} = ( {W^{[l+1]}}^T \delta^{[l+1]} )  \circ g' ( z^{[l]}  )
   $$

3. 最终就可以计算第$l$层的梯度(gradies)：
   
   $$
   \begin{aligned}
    \nabla_{W^{[l]}}J(W,b)&= \delta^{[l]}{a^{[l-1]}  }^T  \\
    \nabla_{b^{[l]}}J(W,b)&= \delta^{[l]}    \\
   \end{aligned}
   $$

上文中的小圆圈负号$\circ$表示元素积(elementwise product)。要注意上面的过程是对应单个训练样本的。

你可以将上面的算法用到逻辑回归(logistic regression)里面（$N=1,g^{[1]}$是S型函数(sigmoid function)$\sigma$）来测试步骤$1$和$3$。还记得$\sigma'(z)=\sigma(z)\circ (1-\sigma(z))$以及$\sigma(z^{[1]})$就是$a^{[1]}$。注意对于逻辑回归的情景，如果$x$是一个实数域内$R^{n\times 1}$的列向量(column vector)，然后$W^{[1]}\in R^{1\times n}$,因此则有$\nabla_{W^{[1]}}J(W,b)\in R^{1\times n}$。代码样本如[http://cs229.stanford.edu/notes/backprop.py](http://cs229.stanford.edu/notes/backprop.py)所示。

（译者注：为了方便我直接把上面链接中的代码贴到下面了。）


```python
#http://cs229.stanford.edu/notes/backprop.py
import numpy as np
from copy import copy

# Example backpropagation code for binary classification with 2-layer
# neural network (single hidden layer)

sigmoid = lambda x: 1 / (1 + np.exp(-x))

def fprop(x, y, params):
  # Follows procedure given in notes
  W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
  z1 = np.dot(W1, x) + b1
  a1 = sigmoid(z1)
  z2 = np.dot(W2, a1) + b2
  a2 = sigmoid(z2)
  loss = -(y * np.log(a2) + (1-y) * np.log(1-a2))
  ret = {'x': x, 'y': y, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'loss': loss}
  for key in params:
    ret[key] = params[key]
  return ret

def bprop(fprop_cache):
  # Follows procedure given in notes
  x, y, z1, a1, z2, a2, loss = [fprop_cache[key] for key in ('x', 'y', 'z1', 'a1', 'z2', 'a2', 'loss')]
  dz2 = (a2 - y)
  dW2 = np.dot(dz2, a1.T)
  db2 = dz2
  dz1 = np.dot(fprop_cache['W2'].T, dz2) * sigmoid(z1) * (1-sigmoid(z1))
  dW1 = np.dot(dz1, x.T)
  db1 = dz1
  return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}

# Gradient checking

if __name__ == '__main__':
  # Initialize random parameters and inputs
  W1 = np.random.rand(2,2)
  b1 = np.random.rand(2, 1)
  W2 = np.random.rand(1, 2)
  b2 = np.random.rand(1, 1)
  params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
  x = np.random.rand(2, 1)
  y = np.random.randint(0, 2)  # Returns 0/1

  fprop_cache = fprop(x, y, params)
  bprop_cache = bprop(fprop_cache)

  # Numerical gradient checking
  # Note how slow this is! Thus we want to use the backpropagation algorithm instead.
  eps = 1e-6
  ng_cache = {}
  # For every single parameter (W, b)
  for key in params:
    param = params[key]
    # This will be our numerical gradient
    ng = np.zeros(param.shape)
    for j in range(ng.shape[0]):
      for k in xrange(ng.shape[1]):
        # For every element of parameter matrix, compute gradient of loss wrt
        # that element numerically using finite differences
        add_eps = np.copy(param)
        min_eps = np.copy(param)
        add_eps[j, k] += eps
        min_eps[j, k] -= eps
        add_params = copy(params)
        min_params = copy(params)
        add_params[key] = add_eps
        min_params[key] = min_eps
        ng[j, k] = (fprop(x, y, add_params)['loss'] - fprop(x, y, min_params)['loss']) / (2 * eps)
    ng_cache[key] = ng

  # Compare numerical gradients to those computed using backpropagation algorithm
  for key in params:
    print key
    # These should be the same
    print(bprop_cache[key])
    print(ng_cache[key])
```
