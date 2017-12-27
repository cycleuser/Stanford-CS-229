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



