# CS229 课程讲义中文翻译
CS229 Lecture notes

|原作者|翻译|
|--|--|
|[Andrew Ng  吴恩达](http://www.andrewng.org/)|[CycleUser](https://www.zhihu.com/people/cycleuser/columns)|

|相关链接|
|--|
|[Github 地址](https://github.com/Kivy-CN/Stanford-CS-229-CN)|
|[知乎专栏](https://zhuanlan.zhihu.com/MachineLearn)|
|[斯坦福大学 CS229 课程网站](http://cs229.stanford.edu/)|
|[网易公开课中文字幕视频](http://open.163.com/movie/2008/1/M/C/M6SGF6VB4_M6SGHFBMC.html)|


# 第一章

## 监督学习（Supervised learning）

咱们先来聊几个使用监督学习来解决问题的实例。假如咱们有一个数据集，里面的数据是俄勒冈州波特兰市的 47 套房屋的面积和价格：

|居住面积（平方英尺）|价格（千美元）|
|--|--|
|2104|400|
|1600|330|
|2400|369|
|1416|232|
|3000|540|
|...|...|
|...|...|
|...|...|

用这些数据来投个图：
![](https://raw.githubusercontent.com/Kivy-CN/Stanford-CS-229-CN/master/img/cs229note1f1.png)

这里要先规范一下符号和含义，这些符号以后还要用到，咱们假设 $x^{(i)}$ 表示 “输入的” 变量值（在这个例子中就是房屋面积），也可以叫做**输入特征**；然后咱们用 $y^{(i)}$ 来表示“输出值”，或者称之为**目标变量**，这个例子里面就是房屋价格。这样的一对 $(x^{(i)},y^{(i)})$就称为一组训练样本，然后咱们用来让机器来学习的数据集，就是一个长度为 m 的训练样本的列表-$\{(x^{(i)},y^{(i)}); i = 1,...,m\}$-也叫做一个**训练集**。另外一定注意，这里的上标 $“^{(i)}”$ 只是作为训练集的索引记号，和数学乘方没有任何关系，千万别误解了。另外我们还会用大写的 $X$ 来表示 **输入值的空间**，大写的 $Y$ 表示** 输出值的空间**。在本节的这个例子中，输入输出的空间都是实数域，所以 $X = Y = R$。
