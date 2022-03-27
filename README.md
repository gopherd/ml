ML: 机器学习算法
================

> 如需查看公式，需要在 chrome 浏览器上安装插件支持:
>
> [https://chrome.google.com/webstore/detail/tex-all-the-things/cbimabofgmfdkicghcadidpemeenbffn/related](https://chrome.google.com/webstore/detail/tex-all-the-things/cbimabofgmfdkicghcadidpemeenbffn/related)

## 1. 决策树（Decision tree）

## 2. 支持向量机（SVM）

### 2.1. 分类器（Classifier）

令 $y = \pmb{w}^{T}\cdot{x} + b$，训练样本 $D = \lbrace({\pmb{x}_i},y_i) | {i\in\lbrace1,...,m\rbrace; y_i\in\lbrace-1,+1\rbrace}\rbrace$，SVM 分类学习将寻求划分平面 $\pmb{w}^{T}\cdot\pmb{x} + b = 0$ 使得平面两侧的间隔最大。

$$
\tag{1}\begin{cases}
\pmb{w}^{T}\cdot\pmb{x} + b \geqslant +1, & y = +1 \newline
\pmb{w}^{T}\cdot\pmb{x} + b \leqslant -1, & y = -1
\end{cases}
$$

SVM 基本型

$$
\tag{2}\min_{\pmb{w},b}\frac{1}{2}\parallel\pmb{w}\parallel^2, st.\space \forall (\pmb{x}, y) \in D, \space y(\pmb{w}^{T}\cdot\pmb{x} + b) \geqslant 1
$$

其拉格朗日形式为

$$
\tag{3}L(\pmb{w}, b, \pmb{\alpha}) = \frac{1}{2}\parallel\pmb{w}\parallel^2+\sum_{i=1}^m{\alpha_i(1 - y_i(\pmb{w}^T\cdot{\pmb{x}_i} + b))}
$$

再由 $\frac{\partial{L}}{\partial{\pmb{w}}} = 0, \frac{\partial{L}}{\partial{b}} = 0$ 可得

$$
\tag{4}\sum_{i=1}^m{\alpha_iy_i\pmb{x}_i} = \pmb{w}
$$

$$
\tag{5}\sum_{i=1}^m\alpha_iy_i = 0
$$

将 (4) 和 (5) 带入 (3) 得到 (2) 的对偶问题

$$
\tag{6}
\max_{\pmb\alpha}\sum_{i=1}^m{\alpha_i} - \frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m{\alpha_i\alpha_jy_iy_j\pmb{x}_i^T\pmb{x}_j}
$$

$$
\tag{7}
st. \sum_{i=1}^m\alpha_iy_i = 0, \space \alpha_i \geqslant 0 \space \forall i \in \lbrace1,...,m\rbrace
$$
