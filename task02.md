## Task02：详读西瓜书+南瓜书第3章——线性回归（3天）
- [一元线性回归：https://www.bilibili.com/video/BV1Mh411e7VU?p=3](https://www.bilibili.com/video/BV1Mh411e7VU?p=3)
- [多元线性回归：https://www.bilibili.com/video/BV1Mh411e7VU?p=4](https://www.bilibili.com/video/BV1Mh411e7VU?p=4)

## 线性模型
### 1.基本形式
- 向量表达法： f(X)=WTX+b, 
- 其中：X =(X1,X2,...XN)代表样本空间N个样本，每个样本x =(x1;x2;...xd)是d维
- 则 X 是一个(N,d)矩阵
- W = (w1;w2;...wd)是学习后得到的权重参数，(d,1)列向量
### 线性回归
- [参考笔记4.2线性回归在机器学习中的应用]([linear](./math/linear.md))
- 总体来说就是试图学得
- f(xi)=wxi+b，使得 f(xi)逼近目标值yi
- 之后转化为求均方误差E = sum(wxi-yi)^2,即求E的最小值
- 均方误差E是一个w二次项函数凸函数，对其进行求导，当dE/dw = 0，找到极值
- 则 dE/dw = d(wx-y)T(wx-y)/dw = 0,得到 xT(xw-y)=0
- xTxw = xTy,当xTx为满秩矩阵才可逆，得
- w = (xTw)'xTy
### 对数线性回归
- 广义线性回归可以理解为用f(x)来逼近y的某种变形
- 线性回归中f(x)逼近y, 对数几率回归是f(x)逼近lny
- 最终 y = e^(wTx+b)，y属于(0,无穷)
### 对数几率回归
- 也是线性回归的一种衍生，适用于做分类任务
- 使得 f(x)逼近 ln(y/1-y)
- 用概率p(y)来代表预估分类为正例的概率，则1-p(y)则判定为负例(这里正例负例是随意命名，可以理解为取{1,0}等)
- 则得到p1(y=1|x) = e^f(x)/1+e^f(x),p0(y=0|x) = 1/1+e^f(x)，构造p(y) = yp1+(1-y)p0,当y=1时,p(y)=p1;当y=0时，p(y) = p0
- 根据极大似然法，构造通用函数l(y) = yln(p1)+ (1-y)ln(p0)满足y=1时, l(y=1)=ln(p1), l(y=0)=ln(p0)
- 通分后为MAX l(y) = yf - ln(1+e^f)
- 转变为求min l(w,b) = -y(wx+b) + ln(1+e^(wx+b))
- 通分一下，求min ln(1+e^f) - ln(e^yf)=ln((1+e^f)/e^yf)，是一个高阶可导连续凸函数
### 线性判别分析
- LDA：给定训练样本集，设法将样例投影到一条直线上，使得同类样例的投影点尽可能接近，异类样例投影点尽可能远离
- 令ui代表不同类的均值向量，sum(i)代表不同类的协方差，则目标函数为
- max J = 类间距离平方/类内样本点协方差
- 得到 J = wT(Sb)W/wT(Sw)W,其中Sb表示类间散度矩阵，Sw表示类内散度矩阵
- 分子分母都是关于w的二次项，wT(Sw)w = 1, 则 目标函数为 Min -wT(Sb)W
- 根据拉格朗日乘子法, wT(Sw)w - 1 = 0,目标函数 Q =wT(sb)w + 入(wT(sw)w - 1)=wT(sb)w + 入wT(sw)w - 入
- 对目标函数进行求导，dQ/dw = -(Sb+SbT)w + 入(Sw+SwT)w, Sb和Sw都为对称矩阵,则dQ/dw = -2Sbw + 2入Sww = 0
- 则 Sbw = 入Sww, 令Sbw = 入(u0-u1), Sww = u0-u1,
- 可对Sw进行奇异值分解 Sw = UsumVT ,Sw可逆, 则 w = Sw'(u0-u1)
### 类别不平衡问题
- 暂略

## 线性代数笔记
[linear](./math/linear.md)