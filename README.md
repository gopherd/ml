brain: 机器学习算法
=================

## 1. 决策树（Decision tree）

## 2. 支持向量机（SVM）

### 2.1. 分类器（Classifier）

令 $y = \bold{w}^{T}\cdot\bold{x} + b$，训练样本 $D = \{(\bold{x_i},y_i) | {i\in\{0,...,m\}; y_i\in\{-1,+1\}}\}$，SVM 分类学习将寻求划分平面 $\bold{w}^{T}\cdot\bold{x} + b = 0$ 使得平面两侧的间隔最大。

```math
\tag{1}\begin{cases}
\bold{w}^{T}\cdot\bold{x} + b \geqslant +1, & y = +1 \\
\bold{w}^{T}\cdot\bold{x} + b \leqslant -1, & y = -1
\end{cases}
```

SVM 基本型

```math
\tag{2}\min_{\bold{w},b}\frac{1}{2}\parallel\bold{w}\parallel^2, st.\space \forall (\bold{x}, y) \in D, \space y(\bold{w}^{T}\cdot\bold{x} + b) \geqslant 1
```

其拉格朗日形式为

```math
\tag{3}L(\bold{w}, b, \bold{\alpha}) = \frac{1}{2}\parallel\bold{w}\parallel^2+\sum_{i=0}^m{\alpha_i(1 - y_i(\bold{w}^T\cdot\bold{x_i} + b))}
```

再由 $\frac{\partial{L}}{\partial{\bold{w}}} = 0, \frac{\partial{L}}{\partial{b}} = 0$ 可得

```math
\tag{4}\sum_{i=0}^m{\alpha_iy_i\bold{x}_i} = \bold{w}
```

```math
\tag{5}\sum_{i=0}^m\alpha_iy_i = 0
```