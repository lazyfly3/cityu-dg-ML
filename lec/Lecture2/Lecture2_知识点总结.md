# Lecture 2 知识点总结 / Knowledge Summary

## Table of Contents / 目录

- [1. Linear Regression / 1. 线性回归](#1-linear-regression--1-线性回归)
- [2. Maximum Likelihood Estimation / 2. 最大似然估计](#2-maximum-likelihood-estimation--2-最大似然估计)
- [3. Gradient Descent / 3. 梯度下降](#3-gradient-descent--3-梯度下降)
- [4. Regularization / 4. 正则化](#4-regularization--4-正则化)

---

## 1. Linear Regression / 1. 线性回归

#### English
Linear regression models the relationship between a dependent variable and one or more independent variables using a linear function. The goal is to find the best-fitting line through the data points.

#### 中文
线性回归使用线性函数建模因变量与一个或多个自变量之间的关系。目标是找到通过数据点的最佳拟合直线。

**数学定义 / Mathematical Definition:**

**线性模型 / Linear Model:**
$$
y = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + \epsilon
$$

**向量形式 / Vector Form:**
$$
y = \mathbf{w}^\top \mathbf{x} + b + \epsilon
$$

其中 / where:
- $y$：目标变量（输出）/ Target variable (output)
- $\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$：特征向量 / Feature vector
- $\mathbf{w} = [w_1, w_2, \ldots, w_n]^\top$：权重向量 / Weight vector
- $b$：偏置项 / Bias term
- $\epsilon$：误差项 / Error term

**目标函数 / Objective Function:**

最小化均方误差 / Minimize mean squared error:

$$
J(\mathbf{w}, b) = \frac{1}{2M} \sum_{i=1}^M (y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b)^2
$$

#### 通俗解释
线性回归就像"画一条最合适的直线"：给你很多点（数据），找一条直线，让所有点到这条直线的距离总和最小。就像用尺子画一条最接近所有点的线。

**计算步骤 / Calculation Steps:**
1. 初始化权重 $\mathbf{w}$ 和偏置 $b$ / Initialize weights $\mathbf{w}$ and bias $b$
2. 计算预测值 $\hat{y}^{(i)} = \mathbf{w}^\top \mathbf{x}^{(i)} + b$ / Calculate predictions
3. 计算误差 $(y^{(i)} - \hat{y}^{(i)})^2$ / Calculate errors
4. 使用梯度下降更新参数 / Update parameters using gradient descent

---

## 2. Maximum Likelihood Estimation / 2. 最大似然估计

#### English
Maximum Likelihood Estimation (MLE) is a method to estimate parameters by maximizing the likelihood function, which measures how likely the observed data is under the model.

#### 中文
最大似然估计（MLE）是一种通过最大化似然函数来估计参数的方法，似然函数衡量在模型下观测数据的可能性。

**似然函数 / Likelihood Function:**

假设误差服从高斯分布 / Assume errors follow Gaussian distribution:

$$
P(y \mid \mathbf{x}, \mathbf{w}, b) = \mathcal{N}(y; \mathbf{w}^\top \mathbf{x} + b, \sigma^2)
$$

**对数似然 / Log-Likelihood:**

$$
\ell(\mathbf{w}, b) = \sum_{i=1}^M \log P(y^{(i)} \mid \mathbf{x}^{(i)}, \mathbf{w}, b)
$$

$$
= -\frac{M}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^M (y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b)^2
$$

**MLE等价于最小化均方误差 / MLE Equivalent to Minimizing MSE:**
最大化对数似然等价于最小化均方误差 / Maximizing log-likelihood is equivalent to minimizing MSE

#### 通俗解释
最大似然估计就像"猜参数"：假设数据是由某个模型生成的，我们找一组参数，使得"看到这些数据"的可能性最大。就像猜硬币的正面概率：如果看到 $10$ 次中 $7$ 次正面，最可能的是概率 $0.7$。

---

## 3. Gradient Descent / 3. 梯度下降

#### English
Gradient descent is an optimization algorithm that iteratively moves in the direction of the steepest descent (negative gradient) to find the minimum of a function.

#### 中文
梯度下降是一种优化算法，通过迭代地向最陡下降方向（负梯度方向）移动来找到函数的最小值。

**更新规则 / Update Rule:**
$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \alpha \nabla_{\mathbf{w}} J(\mathbf{w}^{(t)})
$$

其中 / where:
- $\alpha$：学习率 / Learning rate
- $\nabla_{\mathbf{w}} J$：目标函数对权重的梯度 / Gradient of objective w.r.t. weights

**梯度计算 / Gradient Calculation:**

对线性回归 / For linear regression:

$$
\frac{\partial J}{\partial w_j} = -\frac{1}{M}\sum_{i=1}^M (y^{(i)} - \hat{y}^{(i)}) x_j^{(i)}
$$

$$
\frac{\partial J}{\partial b} = -\frac{1}{M}\sum_{i=1}^M (y^{(i)} - \hat{y}^{(i)})
$$

**计算步骤 / Calculation Steps:**
1. 初始化 $\mathbf{w}$ 和 $b$ / Initialize $\mathbf{w}$ and $b$
2. 计算梯度 $\nabla J$ / Calculate gradient $\nabla J$
3. 更新参数：$\mathbf{w} \leftarrow \mathbf{w} - \alpha \nabla J$ / Update parameters
4. 重复步骤2-3直到收敛 / Repeat steps 2-3 until convergence

#### 通俗解释
梯度下降就像"下山"：想象你在山上，想找到最低点。你每次朝"最陡的下坡方向"走一小步，最终会到达山谷（最小值）。学习率就像"步长"：太大可能跳过最低点，太小走得太慢。

**学习率选择 / Learning Rate Selection:**
- 太小：收敛慢 / Too small: slow convergence
- 太大：可能发散 / Too large: may diverge
- 自适应：使用学习率调度 / Adaptive: use learning rate scheduling

---

## 4. Regularization / 4. 正则化

#### English
Regularization is a technique to prevent overfitting by adding a penalty term to the objective function that discourages large parameter values.

#### 中文
正则化是一种通过向目标函数添加惩罚项来防止过拟合的技术，惩罚项抑制大的参数值。

**L2正则化（Ridge回归）/ L2 Regularization (Ridge Regression):**

**目标函数 / Objective Function:**
$$
J(\mathbf{w}) = \frac{1}{2M}\sum_{i=1}^M (y^{(i)} - \hat{y}^{(i)})^2 + \frac{\lambda}{2}\|\mathbf{w}\|^2
$$

其中 / where:
- $\lambda$：正则化系数 / Regularization coefficient
- $\|\mathbf{w}\|^2 = \sum_j w_j^2$：权重的L2范数平方 / Squared L2 norm of weights

**L1正则化（Lasso回归）/ L1 Regularization (Lasso Regression):**

$$
J(\mathbf{w}) = \frac{1}{2M}\sum_{i=1}^M (y^{(i)} - \hat{y}^{(i)})^2 + \lambda\|\mathbf{w}\|_1
$$

其中 / where:
- $\|\mathbf{w}\|_1 = \sum_j |w_j|$：权重的L1范数 / L1 norm of weights

**区别 / Differences:**
- **L2正则化**：使权重变小但不为零 / L2: shrinks weights but doesn't zero them
- **L1正则化**：可以使某些权重变为零（特征选择）/ L1: can zero out some weights (feature selection)

#### 通俗解释
正则化就像"约束"：不让模型参数变得太大，防止模型"记住"训练数据而不是"学习"规律。L2像"软约束"（让参数小一点），L1像"硬约束"（直接让一些参数变零，相当于删除特征）。

---

## Additional Detail / 补充要点

### Normal Equation / 正规方程

#### English
For linear regression, the optimal solution can be found analytically using the normal equation:

$$
\mathbf{w}^* = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$$

#### 中文
对于线性回归，可以使用正规方程解析地找到最优解：

$$
\mathbf{w}^* = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$$

**适用场景 / When to Use:**
- 特征数量较少时 / When number of features is small
- 需要精确解时 / When exact solution is needed
- 计算 $(\mathbf{X}^\top \mathbf{X})^{-1}$ 可行时 / When computing inverse is feasible

---

### Polynomial Regression / 多项式回归

#### English
Polynomial regression extends linear regression by adding polynomial features, allowing the model to capture nonlinear relationships.

#### 中文
多项式回归通过添加多项式特征扩展线性回归，允许模型捕捉非线性关系。

**多项式特征 / Polynomial Features:**

对于一维输入 / For 1D input:

$$
\phi(x) = [1, x, x^2, x^3, \ldots, x^d]
$$

然后使用线性回归 / Then use linear regression:
$$
y = \mathbf{w}^\top \phi(x) + b
$$

---

### Bias-Variance Tradeoff / 偏差-方差权衡

#### English
- **Bias**: Error from overly simplistic assumptions
- **Variance**: Error from sensitivity to small fluctuations
- **Tradeoff**: Increasing model complexity reduces bias but increases variance

#### 中文
- **偏差**：来自过于简化的假设的误差
- **方差**：来自对小幅波动的敏感性的误差
- **权衡**：增加模型复杂度减少偏差但增加方差

---

## Learning Recommendations / 学习建议

### For Beginners / 对于初学者

1. **Understand the Math / 理解数学:**
   - Matrix operations
   - Derivatives and gradients
   - Optimization basics

2. **Implement from Scratch / 从零实现:**
   - Linear regression
   - Gradient descent
   - Regularization

3. **Visualize Results / 可视化结果:**
   - Plot data and fitted lines
   - Visualize gradient descent
   - Compare with/without regularization

---

## Reference Resources / 参考资源

1. **Textbooks / 教科书:**
   - Pattern Recognition and Machine Learning (Bishop)
   - The Elements of Statistical Learning (Hastie et al.)

2. **Online Resources / 在线资源:**
   - Stanford CS229 Lecture Notes
   - Scikit-learn Documentation

---

## 练习题 / Practice Problems

### 问题1 / Problem 1:
使用梯度下降最小化 $f(x) = x^2 + 2x + 1$，学习率 $\alpha = 0.1$，初始值 $x = 3$。
Use gradient descent to minimize $f(x) = x^2 + 2x + 1$ with learning rate $\alpha = 0.1$, initial value $x = 3$.

### 问题2 / Problem 2:
解释L1和L2正则化的区别，并说明各自的优缺点。
Explain the difference between L1 and L2 regularization, and state their advantages and disadvantages.

### 问题3 / Problem 3:
推导线性回归的闭式解（正规方程）。
Derive the closed-form solution (normal equation) for linear regression.

---

## 例题与解答 / Worked Examples

### 例题1：梯度下降计算 / Gradient Descent Calculation

**题目 / Question:**  
使用梯度下降最小化函数 $f(x) = x^2$，学习率 $\alpha = 0.1$，初始值 $x^{(0)} = 3$。计算前3次迭代。
Use gradient descent to minimize $f(x) = x^2$ with learning rate $\alpha = 0.1$, initial value $x^{(0)} = 3$. Calculate first 3 iterations.

**详细解答 / Detailed Solution:**

**步骤1：计算梯度 / Step 1: Calculate Gradient**
$$
\nabla f(x) = \frac{d}{dx}(x^2) = 2x
$$

**步骤2：第一次迭代 / Step 2: First Iteration**
- 当前值 / Current value: $x^{(0)} = 3$
- 梯度 / Gradient: $\nabla f(3) = 2 \times 3 = 6$
- 更新 / Update: $x^{(1)} = 3 - 0.1 \times 6 = 3 - 0.6 = 2.4$

**步骤3：第二次迭代 / Step 3: Second Iteration**
- 当前值 / Current value: $x^{(1)} = 2.4$
- 梯度 / Gradient: $\nabla f(2.4) = 2 \times 2.4 = 4.8$
- 更新 / Update: $x^{(2)} = 2.4 - 0.1 \times 4.8 = 2.4 - 0.48 = 1.92$

**步骤4：第三次迭代 / Step 4: Third Iteration**
- 当前值 / Current value: $x^{(2)} = 1.92$
- 梯度 / Gradient: $\nabla f(1.92) = 2 \times 1.92 = 3.84$
- 更新 / Update: $x^{(3)} = 1.92 - 0.1 \times 3.84 = 1.92 - 0.384 = 1.536$

**结论 / Conclusion:**
可以看到 $x$ 逐渐接近最小值 $0$。经过3次迭代，$x$ 从 $3$ 减少到 $1.536$。
We can see $x$ gradually approaches the minimum $0$. After 3 iterations, $x$ decreases from $3$ to $1.536$.

---

### 例题2：L2正则化效果 / Effect of L2 Regularization

**题目 / Question:**  
比较有无L2正则化的线性回归目标函数，并解释正则化如何防止过拟合。
Compare linear regression objective functions with and without L2 regularization, and explain how regularization prevents overfitting.

**详细解答 / Detailed Solution:**

**无正则化 / Without Regularization:**
$$
J(\mathbf{w}) = \frac{1}{2M}\sum_{i=1}^M (y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)})^2
$$

**有L2正则化 / With L2 Regularization:**
$$
J(\mathbf{w}) = \frac{1}{2M}\sum_{i=1}^M (y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)})^2 + \frac{\lambda}{2}\|\mathbf{w}\|^2
$$

**正则化的作用 / Role of Regularization:**
1. **惩罚大权重 / Penalizes Large Weights**: 项$\frac{\lambda}{2}\|\mathbf{w}\|^2$鼓励权重变小
2. **减少模型复杂度 / Reduces Model Complexity**: 较小的权重意味着更平滑的决策边界
3. **防止过拟合 / Prevents Overfitting**: 通过限制模型容量，减少对训练数据的过度拟合

**参数$\lambda$的影响 / Effect of Parameter $\lambda$:**
- $\lambda$大：更强的正则化，权重更小，可能欠拟合 / Large $\lambda$: stronger regularization, smaller weights, may underfit
- $\lambda$小：较弱的正则化，权重较大，可能过拟合 / Small $\lambda$: weaker regularization, larger weights, may overfit
- $\lambda=0$：无正则化 / $\lambda=0$: no regularization

---

