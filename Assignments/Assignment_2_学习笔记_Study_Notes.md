# Assignment 2 学习笔记 / Study Notes

## 目录 / Table of Contents
1. [泊松分布与最大似然估计 / Poisson Distribution and MLE](#泊松分布)
2. [梯度下降 / Gradient Descent](#梯度下降)
3. [约束优化与KKT条件 / Constrained Optimization and KKT](#约束优化)
4. [支持向量机 / Support Vector Machine](#支持向量机)

---

## 泊松分布 / Poisson Distribution

### 1.1 定义与性质 / Definition and Properties

**中文解释：**
泊松分布描述在固定时间或空间间隔内发生的事件数量，假设事件以恒定速率独立发生。

**English Explanation:**
Poisson distribution describes the number of events occurring in a fixed interval of time or space, assuming events occur at a constant rate and independently.

**概率质量函数 / Probability Mass Function:**  
$$
P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}
$$

**符号说明 / Symbol Explanation:**
- $P(X=k)$：随机变量X取值为k的概率 / Probability that random variable X equals k
- $k$：事件发生的次数，取值为0, 1, 2, ... / Number of occurrences, values: 0, 1, 2, ...
- $\lambda$：泊松分布的参数，表示平均事件数（必须>0）/ Parameter of Poisson distribution, average number of events (must be >0)
- $e$：自然常数，约等于2.718 / Euler's number, approximately 2.718
- $e^{-\lambda}$：e的负λ次方 / e raised to the power of -λ
- $\lambda^k$：λ的k次方 / λ raised to the power of k
- $k!$：k的阶乘，k! = k × (k-1) × ... × 2 × 1 / Factorial of k, k! = k × (k-1) × ... × 2 × 1

**计算步骤 / Calculation Steps:**
1. 计算 $\lambda^k$（λ的k次方）/ Calculate $\lambda^k$ (λ to the power of k)
2. 计算 $e^{-\lambda}$（e的负λ次方）/ Calculate $e^{-\lambda}$ (e to the power of -λ)
3. 计算 $k!$（k的阶乘）/ Calculate $k!$ (factorial of k)
4. 将步骤1和2的结果相乘，再除以步骤3的结果 / Multiply results from steps 1 and 2, then divide by result from step 3

**计算示例 / Calculation Example:**
假设λ=2，计算k=3时的概率（即事件发生3次的概率）
Suppose λ=2, calculate probability when k=3 (probability of 3 events occurring)

步骤1: λ^k = 2^3 = 8
步骤2: e^(-λ) = e^(-2) ≈ 0.1353
步骤3: k! = 3! = 3×2×1 = 6
步骤4: P(X=3) = (8 × 0.1353) / 6 = 1.0824 / 6 ≈ 0.1804

所以当平均事件数为2时，发生3次事件的概率约为18.04%
So when average number of events is 2, probability of 3 events is approximately 18.04%

再计算k=0（无事件发生）:
Step 1: 2^0 = 1
Step 2: e^(-2) ≈ 0.1353
Step 3: 0! = 1
Step 4: P(X=0) = (1 × 0.1353) / 1 = 0.1353 ≈ 13.53%

**重要性质 / Important Properties:**
- 均值 / Mean: E[X] = λ
- 方差 / Variance: Var(X) = λ
- 注意：均值和方差相等！/ Note: Mean equals variance!

### 1.2 最大似然估计 / Maximum Likelihood Estimation

**似然函数 / Likelihood Function:**
给定独立同分布样本 D = {k(1), k(2), ..., k(M)}:
Given i.i.d. samples D = {k(1), k(2), ..., k(M)}:

$$
L(\lambda) = \prod_{i=1}^M \frac{\lambda^{k^{(i)}} e^{-\lambda}}{k^{(i)}!}
$$

**对数似然 / Log-Likelihood:**
$$
\ell(\lambda) = \log L(\lambda) = \sum_{i=1}^M \big[k^{(i)} \log \lambda - \lambda - \log(k^{(i)}!)\big]
$$

**符号说明 / Symbol Explanation:**
- $\ell(\lambda)$：对数似然函数 / Log-likelihood function
- $\log L(\lambda)$：似然函数的自然对数 / Natural logarithm of likelihood function
- $\sum_{i=1}^M$：对所有M个样本求和 / Sum over all M samples
- $k^{(i)}$：第i个样本中事件发生的次数 / Number of events in i-th sample
- $\log \lambda$：λ的自然对数 / Natural logarithm of λ
- $-\lambda$：负的λ值 / Negative of λ
- $\log(k^{(i)}!)$：k^(i)的阶乘的对数（常数项，与λ无关）/ Logarithm of factorial of k^(i) (constant term, independent of λ)

**计算步骤 / Calculation Steps:**
1. 对每个样本i，计算 $k^{(i)} \log \lambda - \lambda - \log(k^{(i)}!)$ / For each sample i, calculate $k^{(i)} \log \lambda - \lambda - \log(k^{(i)}!)$
2. 将所有样本的结果相加 / Sum results from all samples
3. 注意：$\log(k^{(i)}!)$是常数，在求导时会消失 / Note: $\log(k^{(i)}!)$ is constant, disappears when taking derivative

**求导并令其为零 / Taking Derivative and Setting to Zero:**
$$
\frac{\partial \ell}{\partial \lambda} = \sum_{i=1}^M \left(\frac{k^{(i)}}{\lambda} - 1\right) = 0
$$

**符号说明 / Symbol Explanation:**
- $\frac{\partial \ell}{\partial \lambda}$：对数似然函数对参数λ的偏导数 / Partial derivative of log-likelihood with respect to parameter λ
- $\frac{k^{(i)}}{\lambda}$：来自项$k^{(i)} \log \lambda$的导数（$\frac{d}{d\lambda} k^{(i)} \log \lambda = \frac{k^{(i)}}{\lambda}$）/ Derivative from term $k^{(i)} \log \lambda$ ($\frac{d}{d\lambda} k^{(i)} \log \lambda = \frac{k^{(i)}}{\lambda}$)
- $-1$：来自项$-\lambda$的导数（$\frac{d}{d\lambda}(-\lambda) = -1$）/ Derivative from term $-\lambda$ ($\frac{d}{d\lambda}(-\lambda) = -1$)
- $= 0$：在最优解处，导数必须为零 / At optimum, derivative must be zero

**计算步骤 / Calculation Steps:**
1. 对每个样本i，计算 $\frac{k^{(i)}}{\lambda} - 1$ / For each sample i, calculate $\frac{k^{(i)}}{\lambda} - 1$
2. 将所有样本的结果相加并令其等于0 / Sum results from all samples and set equal to 0
3. 得到：$\frac{1}{\lambda}\sum_{i=1}^M k^{(i)} - M = 0$ / Get: $\frac{1}{\lambda}\sum_{i=1}^M k^{(i)} - M = 0$

**最大似然估计 / MLE:**
$$
\lambda^* = \frac{1}{M}\sum_{i=1}^M k^{(i)} \quad\text{(样本均值 / sample mean)}
$$

**计算步骤 / Calculation Steps:**
1. 从导数方程：$\frac{1}{\lambda}\sum_{i=1}^M k^{(i)} = M$ / From derivative equation: $\frac{1}{\lambda}\sum_{i=1}^M k^{(i)} = M$
2. 两边同时乘以λ：$\sum_{i=1}^M k^{(i)} = M\lambda$ / Multiply both sides by λ: $\sum_{i=1}^M k^{(i)} = M\lambda$
3. 两边同时除以M：$\lambda^* = \frac{1}{M}\sum_{i=1}^M k^{(i)}$ / Divide both sides by M: $\lambda^* = \frac{1}{M}\sum_{i=1}^M k^{(i)}$
4. 结果：λ的最大似然估计就是所有样本的平均值 / Result: MLE of λ is the sample mean

**计算示例 / Calculation Example:**
假设观察到5个时间间隔，每个间隔的事件数为：{2, 0, 3, 1, 2}
Suppose observe 5 time intervals with event counts: {2, 0, 3, 1, 2}

计算最大似然估计 / Calculate MLE:
- M = 5（样本数）/ M = 5 (number of samples)
- $\sum_{i=1}^5 k^{(i)} = 2 + 0 + 3 + 1 + 2 = 8$
- $\lambda^* = \frac{8}{5} = 1.6$

所以λ的最大似然估计为1.6，即平均每个时间间隔发生1.6个事件
So MLE of λ is 1.6, meaning average 1.6 events per time interval

**中文解释：**
泊松分布参数λ的最大似然估计就是样本的平均值。

**English Explanation:**
The MLE of λ for Poisson distribution is simply the sample mean.

### 1.3 实际应用示例 / Practical Example

**问题 / Problem:**
观察230个时间间隔，统计每个间隔的事件发生次数：
Observe 230 time intervals, count events per interval:

| 发生次数 k | 0 | 1 | 2 | 3 | 4+ |
| Occurrences k | 0 | 1 | 2 | 3 | 4+ |
| 间隔数 | 100 | 81 | 34 | 9 | 6 |
| Intervals | 100 | 81 | 34 | 9 | 6 |

**计算 / Calculation:**
$$
\lambda^* = \frac{0\times100 + 1\times81 + 2\times34 + 3\times9 + 4\times6}{230}
= \frac{200}{230} \approx 0.87
$$

---

## 梯度下降 / Gradient Descent

### 2.1 基本思想 / Basic Idea

**中文解释：**
梯度下降是一种优化算法，通过沿着目标函数梯度的反方向迭代更新参数，逐步找到最小值。

**English Explanation:**
Gradient descent is an optimization algorithm that iteratively updates parameters in the direction opposite to the gradient, gradually finding the minimum.

**更新规则 / Update Rule:**  
$$
\theta^{(t+1)} = \theta^{(t)} - \alpha\, \nabla f\big(\theta^{(t)}\big)
$$

**符号说明 / Symbol Explanation:**
- $\theta^{(t)}$：第t次迭代时的参数值 / Parameter values at iteration t
- $\theta^{(t+1)}$：第t+1次迭代时的参数值（更新后）/ Parameter values at iteration t+1 (after update)
- $\alpha$：学习率，控制每次更新的步长（通常0 < α < 1）/ Learning rate, controls step size (usually 0 < α < 1)
- $\nabla f(\theta^{(t)})$：目标函数f在$\theta^{(t)}$处的梯度（偏导数向量）/ Gradient of objective function f at $\theta^{(t)}$ (vector of partial derivatives)
- 减号：沿着梯度反方向更新（因为要最小化函数）/ Minus sign: update in direction opposite to gradient (to minimize function)

**计算步骤 / Calculation Steps:**
1. 计算当前参数$\theta^{(t)}$处的梯度$\nabla f(\theta^{(t)})$ / Calculate gradient $\nabla f(\theta^{(t)})$ at current parameters
2. 将梯度乘以学习率α / Multiply gradient by learning rate α
3. 从当前参数中减去步骤2的结果 / Subtract result from step 2 from current parameters
4. 得到新的参数值$\theta^{(t+1)}$ / Obtain new parameter values $\theta^{(t+1)}$

**计算示例 / Calculation Example:**
假设要最小化 f(x) = x²，初始值x^(0) = 3，学习率α = 0.1
Suppose minimize f(x) = x², initial value x^(0) = 3, learning rate α = 0.1

第1次迭代 / First iteration:
- 梯度: ∇f = 2x = 2×3 = 6
- 更新: x^(1) = 3 - 0.1×6 = 3 - 0.6 = 2.4

第2次迭代 / Second iteration:
- 梯度: ∇f = 2×2.4 = 4.8
- 更新: x^(2) = 2.4 - 0.1×4.8 = 2.4 - 0.48 = 1.92

第3次迭代 / Third iteration:
- 梯度: ∇f = 2×1.92 = 3.84
- 更新: x^(3) = 1.92 - 0.1×3.84 = 1.92 - 0.384 = 1.536

可以看到x逐渐接近最小值0
Can see x gradually approaches minimum 0

### 2.2 偏导数计算 / Partial Derivative Calculation

**问题 / Problem:** 最小化误差函数 / Minimize error function:  
$$
l(u,v) = \big(u e^{v} - 2v e^{-u}\big)^2
$$

**对u的偏导数 / Partial Derivative w.r.t. u:**  
$$
\frac{\partial l}{\partial u} = 2\big(u e^{v} - 2v e^{-u}\big)\big(e^{v} + 2v e^{-u}\big)
$$

**符号说明 / Symbol Explanation:**
- $\frac{\partial l}{\partial u}$：误差函数l对变量u的偏导数 / Partial derivative of error function l with respect to u
- $2$：来自链式法则的系数（对平方函数求导）/ Coefficient from chain rule (derivative of square function)
- $u e^{v} - 2v e^{-u}$：原函数f(u,v)的值 / Value of original function f(u,v)
- $e^{v} + 2v e^{-u}$：f(u,v)对u的偏导数 / Partial derivative of f(u,v) with respect to u
- $e^{v}$：e的v次方对u求导为0，但作为f的一部分保留 / e^v has derivative 0 w.r.t. u, but kept as part of f
- $2v e^{-u}$：对u求导得到 $2v e^{-u}$（注意负号）/ Derivative w.r.t. u gives $2v e^{-u}$ (note the negative sign)

**计算步骤 / Calculation Steps:**
1. 识别函数形式：$l = [f(u,v)]^2$，其中 $f(u,v) = u e^{v} - 2v e^{-u}$ / Identify form: $l = [f(u,v)]^2$ where $f(u,v) = u e^{v} - 2v e^{-u}$
2. 应用链式法则：$\frac{\partial l}{\partial u} = 2f \cdot \frac{\partial f}{\partial u}$ / Apply chain rule: $\frac{\partial l}{\partial u} = 2f \cdot \frac{\partial f}{\partial u}$
3. 计算$\frac{\partial f}{\partial u} = e^{v} + 2v e^{-u}$ / Calculate $\frac{\partial f}{\partial u} = e^{v} + 2v e^{-u}$
4. 代入得到最终结果 / Substitute to get final result

**计算示例 / Calculation Example:**
假设u=1, v=1，计算$\frac{\partial l}{\partial u}$
Suppose u=1, v=1, calculate $\frac{\partial l}{\partial u}$

步骤1: 计算f(1,1) = 1×e^1 - 2×1×e^(-1) = e - 2/e ≈ 2.718 - 0.736 = 1.982
步骤2: 计算$\frac{\partial f}{\partial u}$ = e^1 + 2×1×e^(-1) = e + 2/e ≈ 2.718 + 0.736 = 3.454
步骤3: $\frac{\partial l}{\partial u}$ = 2×1.982×3.454 ≈ 13.69

类似地计算$\frac{\partial l}{\partial v}$:
Similarly calculate $\frac{\partial l}{\partial v}$:
- $\frac{\partial f}{\partial v}$ = 1×e^1 - 2×e^(-1) = e - 2/e ≈ 1.982
- $\frac{\partial l}{\partial v}$ = 2×1.982×1.982 ≈ 7.86

**对v的偏导数 / Partial Derivative w.r.t. v:**  
$$
\frac{\partial l}{\partial v} = 2\big(u e^{v} - 2v e^{-u}\big)\big(u e^{v} - 2 e^{-u}\big)
$$

**符号说明 / Symbol Explanation:**
- $\frac{\partial l}{\partial v}$：误差函数l对变量v的偏导数 / Partial derivative of error function l with respect to v
- $u e^{v} - 2 e^{-u}$：f(u,v)对v的偏导数 / Partial derivative of f(u,v) with respect to v
- $u e^{v}$：对v求导得到 $u e^{v}$ / Derivative w.r.t. v gives $u e^{v}$
- $-2 e^{-u}$：对v求导，$e^{-u}$的导数为0（因为不含v）/ Derivative w.r.t. v, derivative of $e^{-u}$ is 0 (no v)

**计算步骤 / Calculation Steps:**
1. 应用链式法则：$\frac{\partial l}{\partial v} = 2f \cdot \frac{\partial f}{\partial v}$ / Apply chain rule: $\frac{\partial l}{\partial v} = 2f \cdot \frac{\partial f}{\partial v}$
2. 计算$\frac{\partial f}{\partial v} = u e^{v} - 2 e^{-u}$ / Calculate $\frac{\partial f}{\partial v} = u e^{v} - 2 e^{-u}$
3. 代入得到最终结果 / Substitute to get final result

**链式法则 / Chain Rule:**
使用链式法则：$\frac{d}{dx} [f(x)]^2 = 2f(x) \cdot f'(x)$
Using chain rule: $\frac{d}{dx} [f(x)]^2 = 2f(x) \cdot f'(x)$

### 2.3 算法实现 / Algorithm Implementation

**Python伪代码 / Python Pseudocode:**
```python
def gradient_descent(f, partial_derivatives, n_variables, 
                     lr=0.1, max_iter=20, tolerance=1e-14):
    theta = [1, 1]  # 初始值 / initial values
    y_cur = f(*theta)
    
    for i in range(max_iter):
        # 计算梯度 / Calculate gradient
        gradient = [df(*theta) for df in partial_derivatives]
        
        # 更新参数 / Update parameters
        for j in range(n_variables):
            theta[j] -= gradient[j] * lr
        
        # 检查收敛 / Check convergence
        y_cur, y_pre = f(*theta), y_cur
        if y_cur < tolerance:
            print(i + 1)  # 迭代次数 / number of iterations
            break
    
    return theta
```

**关键点 / Key Points:**
- 学习率α控制步长 / Learning rate α controls step size
- 太小：收敛慢 / Too small: slow convergence
- 太大：可能发散 / Too large: may diverge
- 需要合适的停止条件 / Need appropriate stopping condition

---

## 约束优化 / Constrained Optimization

### 3.1 拉格朗日乘数法 / Lagrange Multipliers

**中文解释：**
拉格朗日乘数法用于求解带等式约束的优化问题。

**English Explanation:**
Lagrange multipliers method solves optimization problems with equality constraints.

**拉格朗日函数 / Lagrangian Function:**  
$$
L(x,\lambda) = f(x) + \sum_i \lambda_i\, h_i(x)
$$

其中 / where:
- f(x) 是目标函数 / f(x) is objective function
- h_i(x) = 0 是约束条件 / h_i(x) = 0 are constraints
- λ_i 是拉格朗日乘数 / λ_i are Lagrange multipliers

### 3.2 KKT条件 / KKT Conditions

**中文解释：**
KKT条件是约束优化问题最优解的必要条件，推广了拉格朗日乘数法到不等式约束。

**English Explanation:**
KKT conditions are necessary conditions for optimality in constrained optimization, extending Lagrange multipliers to inequality constraints.

**KKT条件包括 / KKT Conditions Include:**

1. **平稳性条件 / Stationarity:**  
   $$\nabla_x L = 0$$
   - $\nabla_x L$：拉格朗日函数L对变量x的梯度 / Gradient of Lagrangian L with respect to variables x
   - $= 0$：梯度必须为零（在最优解处）/ Gradient must be zero (at optimal solution)
   - 含义：在最优解处，目标函数的梯度等于约束梯度的线性组合 / Meaning: at optimum, gradient of objective equals linear combination of constraint gradients

2. **原始可行性 / Primal Feasibility:**  
   $$g_i(x) \le 0,\quad h_j(x)=0$$
   - $g_i(x) \le 0$：不等式约束必须满足（≤0形式）/ Inequality constraints must be satisfied (in ≤0 form)
   - $h_j(x)=0$：等式约束必须满足 / Equality constraints must be satisfied
   - 含义：最优解必须在可行域内 / Meaning: optimal solution must be in feasible region

3. **对偶可行性 / Dual Feasibility:**  
   $$\lambda_i \ge 0$$
   - $\lambda_i$：不等式约束$g_i(x) \le 0$对应的拉格朗日乘数 / Lagrange multiplier for inequality constraint $g_i(x) \le 0$
   - $\ge 0$：不等式约束的乘数必须非负 / Multipliers for inequality constraints must be non-negative
   - 含义：只有"起作用"的约束（$g_i(x)=0$）才可能有正的乘数 / Meaning: only "active" constraints ($g_i(x)=0$) can have positive multipliers

4. **互补松弛性 / Complementary Slackness:**  
   $$\lambda_i\, g_i(x) = 0$$
   - $\lambda_i \times g_i(x) = 0$：拉格朗日乘数与约束值的乘积必须为零 / Product of Lagrange multiplier and constraint value must be zero
   - 含义：如果约束不活跃（$g_i(x) < 0$），则$\lambda_i = 0$；如果$\lambda_i > 0$，则约束必须活跃（$g_i(x) = 0$）/ Meaning: if constraint inactive ($g_i(x) < 0$), then $\lambda_i = 0$; if $\lambda_i > 0$, constraint must be active ($g_i(x) = 0$)

### 3.3 示例问题 / Example Problem

**优化问题 / Optimization Problem:**  
$$
\begin{aligned}
\min\; &f(x_1,x_2)= x_1^2 + x_2^2 + 2x_1 + 3x_2 \\
\text{s.t. }& x_1 + x_2 \ge 1,\;\; x_1 - 2x_2 \le 2,\;\; x_1 \ge 0
\end{aligned}
$$

**拉格朗日函数 / Lagrangian:**  
$$
L = x_1^2 + x_2^2 + 2x_1 + 3x_2
  + \lambda_1(1 - x_1 - x_2)
  + \lambda_2(x_1 - 2x_2 - 2)
  + \lambda_3(-x_1)
$$

**求解步骤 / Solution Steps:**

1. **无约束最小值 / Unconstrained Minimum:**  
   $$\nabla f=0 \Rightarrow x_1=-2,\; x_2=-1.5$$
   但不可行（违反约束）/ But infeasible (violates constraints)

2. **检查约束激活情况 / Check Active Constraints:**
   - 如果第一个约束激活 / If first constraint active: x₁ + x₂ = 1
   - 求解KKT条件 / Solve KKT conditions
   - 得到 / Get: x₁ = 1, x₂ = 0, λ₁ = 3

3. **验证 / Verification:**
   - 满足所有约束 / Satisfies all constraints
   - 目标值 / Objective value: f(1, 0) = 2.5

---

## 支持向量机 / Support Vector Machine

### 4.1 基本概念 / Basic Concepts

**中文解释：**
SVM寻找最优分离超平面，使得两类数据之间的间隔（margin）最大。

**English Explanation:**
SVM finds the optimal separating hyperplane that maximizes the margin between two classes.

**超平面方程 / Hyperplane Equation:**  
$$
w^\top x + b = 0
$$

**符号说明 / Symbol Explanation:**
- $w$：超平面的法向量（权重向量），决定超平面的方向 / Normal vector (weight vector) of hyperplane, determines direction
- $w^\top$：w的转置（行向量形式）/ Transpose of w (row vector form)
- $x$：空间中的一个点（特征向量）/ A point in space (feature vector)
- $w^\top x$：向量w和x的内积（点积），等于 $\sum_i w_i x_i$ / Inner product (dot product) of vectors w and x, equals $\sum_i w_i x_i$
- $b$：偏置项（截距），决定超平面的位置 / Bias term (intercept), determines position of hyperplane
- $= 0$：超平面上的点满足此等式 / Points on hyperplane satisfy this equation

**计算步骤 / Calculation Steps:**
1. 计算 $w^\top x$（将w和x对应元素相乘后求和）/ Calculate $w^\top x$ (multiply corresponding elements and sum)
2. 加上偏置b / Add bias b
3. 如果结果等于0，点在超平面上；>0在一侧，<0在另一侧 / If result equals 0, point is on hyperplane; >0 on one side, <0 on other side

**计算示例 / Calculation Example:**
假设w = [2, -1], b = 3，判断点x = [1, 5]在超平面的哪一侧
Suppose w = [2, -1], b = 3, determine which side of hyperplane point x = [1, 5] is on

步骤1: w^T x = 2×1 + (-1)×5 = 2 - 5 = -3
步骤2: w^T x + b = -3 + 3 = 0

结果等于0，所以点x在超平面上
Result equals 0, so point x is on the hyperplane

再判断点x = [2, 1]:
Step 1: w^T x = 2×2 + (-1)×1 = 4 - 1 = 3
Step 2: w^T x + b = 3 + 3 = 6 > 0

结果>0，点在超平面的一侧（正侧）
Result > 0, point is on one side (positive side) of hyperplane

**间隔 / Margin:**
点到超平面的距离 / Distance from point to hyperplane

### 4.2 点到超平面的距离 / Distance from Point to Hyperplane

**优化问题 / Optimization Problem:**  
$$
\min_x \|x^{(i)} - x\|^2 \quad \text{s.t. } w^\top x + b = 0
$$

**拉格朗日函数 / Lagrangian:**  
$$
L(x,\lambda) = \|x^{(i)} - x\|^2 + \lambda (w^\top x + b)
$$

**求解 / Solution:**
1. 对x求导并令其为零 / Take derivative w.r.t. x and set to zero:  
   $$x = x^{(i)} - \lambda w$$

2. 代入约束 / Substitute into constraint:  
   $$w^\top(x^{(i)} - \lambda w) + b = 0 \;\Rightarrow\; \lambda = \frac{w^\top x^{(i)} + b}{\|w\|^2}$$

3. 计算距离 / Calculate distance:  
   $$\operatorname{dist} = \|x^{(i)} - x\| = \frac{|w^\top x^{(i)} + b|}{\|w\|}$$

**符号说明 / Symbol Explanation:**
- $\operatorname{dist}$：点到超平面的距离 / Distance from point to hyperplane
- $x^{(i)}$：第i个数据点 / i-th data point
- $x$：超平面上离$x^{(i)}$最近的点 / Point on hyperplane closest to $x^{(i)}$
- $\|x^{(i)} - x\|$：两点之间的欧氏距离 / Euclidean distance between two points
- $|w^\top x^{(i)} + b|$：点到超平面的"未归一化距离"（绝对值）/ "Unnormalized distance" from point to hyperplane (absolute value)
- $\|w\|$：权重向量w的模长（L2范数），$\|w\| = \sqrt{\sum_i w_i^2}$ / Magnitude (L2 norm) of weight vector w, $\|w\| = \sqrt{\sum_i w_i^2}$
- 除以$\|w\|$：将距离归一化 / Dividing by $\|w\|$: normalizes the distance

**计算步骤 / Calculation Steps:**
1. 计算 $w^\top x^{(i)} + b$（点到超平面的"原始距离"）/ Calculate $w^\top x^{(i)} + b$ ("raw distance" from point to hyperplane)
2. 取绝对值 $|w^\top x^{(i)} + b|$ / Take absolute value $|w^\top x^{(i)} + b|$
3. 计算权重向量的模长 $\|w\| = \sqrt{w_1^2 + w_2^2 + ... + w_n^2}$ / Calculate magnitude of weight vector $\|w\| = \sqrt{w_1^2 + w_2^2 + ... + w_n^2}$
4. 将步骤2的结果除以步骤3的结果 / Divide result from step 2 by result from step 3

**计算示例 / Calculation Example:**
假设w = [3, 4], b = -5，计算点x = [1, 2]到超平面的距离
Suppose w = [3, 4], b = -5, calculate distance from point x = [1, 2] to hyperplane

步骤1: w^T x + b = 3×1 + 4×2 - 5 = 3 + 8 - 5 = 6
步骤2: |6| = 6
步骤3: ||w|| = √(3² + 4²) = √(9 + 16) = √25 = 5
步骤4: 距离 = 6 / 5 = 1.2

所以点x到超平面的距离为1.2个单位
So distance from point x to hyperplane is 1.2 units

### 4.3 硬间隔SVM / Hard Margin SVM

**原始问题 / Primal Problem:**  
$$
\min \tfrac{1}{2}\|w\|^2 \quad
\text{s.t. } y^{(i)}(w^\top x^{(i)} + b) \ge 1,\; i=1,\dots,M
$$

**符号说明 / Symbol Explanation:**
- $\min$：最小化目标函数 / Minimize objective function
- $\tfrac{1}{2}\|w\|^2$：目标函数，最小化权重向量的模长的平方 / Objective function, minimize squared magnitude of weight vector
- $\|w\|^2$：权重向量w的L2范数的平方，等于$\sum_j w_j^2$ / Squared L2 norm of weight vector w, equals $\sum_j w_j^2$
- $\tfrac{1}{2}$：系数，使后续求导更方便 / Coefficient, makes subsequent derivatives easier
- $\text{s.t.}$：subject to，约束条件 / Subject to, constraints
- $y^{(i)}$：第i个样本的标签，取值为+1或-1 / Label of i-th sample, value +1 or -1
- $w^\top x^{(i)} + b$：第i个样本到超平面的"原始距离" / "Raw distance" of i-th sample to hyperplane
- $y^{(i)}(w^\top x^{(i)} + b) \ge 1$：约束条件，要求所有样本到超平面的"函数间隔"至少为1 / Constraint: all samples must have "functional margin" at least 1
- $M$：训练样本的数量 / Number of training samples

**计算步骤 / Calculation Steps:**
1. 初始化权重向量w和偏置b / Initialize weight vector w and bias b
2. 检查所有样本是否满足约束：$y^{(i)}(w^\top x^{(i)} + b) \ge 1$ / Check if all samples satisfy constraint: $y^{(i)}(w^\top x^{(i)} + b) \ge 1$
3. 如果不满足，调整w和b使约束满足 / If not satisfied, adjust w and b to satisfy constraints
4. 在满足约束的前提下，最小化$\tfrac{1}{2}\|w\|^2$ / Minimize $\tfrac{1}{2}\|w\|^2$ while satisfying constraints
5. 使用拉格朗日乘数法或对偶方法求解 / Use Lagrange multipliers or dual method to solve

**计算示例 / Calculation Example:**
假设有2个样本，x^(1) = [1, 1], y^(1) = +1; x^(2) = [2, 2], y^(2) = -1
Suppose 2 samples: x^(1) = [1, 1], y^(1) = +1; x^(2) = [2, 2], y^(2) = -1

假设找到最优解w = [1, -1], b = 0
Suppose optimal solution is w = [1, -1], b = 0

检查约束 / Check constraints:
- 样本1: y^(1)(w^T x^(1) + b) = +1 × (1×1 + (-1)×1 + 0) = +1 × 0 = 0 < 1 ✗
- 样本2: y^(2)(w^T x^(2) + b) = -1 × (1×2 + (-1)×2 + 0) = -1 × 0 = 0 < 1 ✗

这个解不满足约束，需要调整
This solution doesn't satisfy constraints, need to adjust

调整后假设w = [2, -2], b = 1:
After adjustment, suppose w = [2, -2], b = 1:
- 样本1: +1 × (2×1 + (-2)×1 + 1) = +1 × 1 = 1 ≥ 1 ✓
- 样本2: -1 × (2×2 + (-2)×2 + 1) = -1 × 1 = -1 < 1 ✗

仍需要进一步优化
Still need further optimization

**拉格朗日函数 / Lagrangian:**  
$$
L(w,b,\alpha) = \tfrac{1}{2}\|w\|^2 - \sum_i \alpha_i\big[y^{(i)}(w^\top x^{(i)} + b) - 1\big]
$$

**对偶问题 / Dual Problem:**  
$$
\begin{aligned}
\max_{\alpha}\;& \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j} \alpha_i \alpha_j y^{(i)}y^{(j)} (x^{(i)})^\top x^{(j)} \\
\text{s.t. } & \sum_i \alpha_i y^{(i)} = 0,\;\; \alpha_i \ge 0
\end{aligned}
$$

### 4.4 软间隔SVM / Soft Margin SVM

**原始问题 / Primal Problem:**  
$$
\min_{w,b,\xi}\; \tfrac{1}{2}\|w\|^2 + C\sum_i \xi_i \quad
\text{s.t. } y^{(i)}(w^\top x^{(i)} + b) \ge 1 - \xi_i,\;\; \xi_i \ge 0
$$

**对偶问题 / Dual Problem:**  
$$
\begin{aligned}
\max_{\alpha}\;& \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j} \alpha_i \alpha_j y^{(i)}y^{(j)} (x^{(i)})^\top x^{(j)} \\
\text{s.t. } & \sum_i \alpha_i y^{(i)} = 0,\;\; 0 \le \alpha_i \le C
\end{aligned}
$$

**关键区别 / Key Difference:**
- 硬间隔：α_i ≥ 0 / Hard margin: α_i ≥ 0
- 软间隔：0 ≤ α_i ≤ C / Soft margin: 0 ≤ α_i ≤ C

**参数C的作用 / Role of Parameter C:**
- C大：更少误分类，但可能过拟合 / Large C: fewer misclassifications, but may overfit
- C小：允许更多误分类，更平滑的决策边界 / Small C: allow more misclassifications, smoother decision boundary

---

## 学习建议 / Study Recommendations

### 对于初学者 / For Beginners:

1. **理解优化基础 / Understand Optimization Basics:**
   - 梯度、方向导数 / Gradient, directional derivative
   - 凸优化概念 / Convex optimization concepts

2. **掌握KKT条件 / Master KKT Conditions:**
   - 从拉格朗日乘数法开始 / Start with Lagrange multipliers
   - 理解互补松弛性 / Understand complementary slackness

3. **SVM的几何直觉 / Geometric Intuition for SVM:**
   - 可视化超平面和间隔 / Visualize hyperplane and margin
   - 理解支持向量的作用 / Understand role of support vectors

4. **编程实践 / Programming Practice:**
   - 实现梯度下降 / Implement gradient descent
   - 实现简单SVM / Implement simple SVM
   - 可视化优化过程 / Visualize optimization process

### 常见错误 / Common Mistakes:

1. **梯度方向错误 / Wrong gradient direction**
   - 记住：梯度指向函数值增加最快的方向 / Remember: gradient points to direction of fastest increase
   - 下降需要负梯度 / Descent needs negative gradient

2. **KKT条件应用错误 / Incorrect KKT application**
   - 忘记互补松弛性 / Forgetting complementary slackness
   - 混淆等式和不等式约束 / Confusing equality and inequality constraints

3. **SVM对偶形式推导错误 / Errors in SVM dual derivation**
   - 注意符号 / Pay attention to signs
   - 正确应用拉格朗日对偶 / Correctly apply Lagrangian duality

---

## 练习题 / Practice Problems

### 问题1 / Problem 1:
使用梯度下降最小化 f(x) = x² + 2x + 1，学习率α = 0.1
Use gradient descent to minimize f(x) = x² + 2x + 1 with learning rate α = 0.1

### 问题2 / Problem 2:
求解约束优化问题：
Solve constrained optimization:
$$
\min_{x,y} x^2 + y^2 \quad \text{s.t. } x + y = 1
$$

### 问题3 / Problem 3:
推导软间隔SVM的对偶形式
Derive the dual form of soft margin SVM

---

## 参考资源 / Reference Resources

1. **优化理论 / Optimization Theory:**
   - Convex Optimization (Boyd & Vandenberghe)
   - Numerical Optimization (Nocedal & Wright)

2. **支持向量机 / Support Vector Machine:**
   - The Elements of Statistical Learning (Hastie et al.)
   - Pattern Recognition and Machine Learning (Bishop)

3. **在线课程 / Online Courses:**
   - Stanford CS229: Machine Learning
   - MIT 6.034: Artificial Intelligence

---

## 例题与解答 / Worked Examples

### 例题1：泊松分布的最大似然估计 / MLE for Poisson Distribution

**题目 / Question (a):**  
泊松分布是一个离散概率分布，表示在固定时间或空间间隔内发生的事件数量。设x服从泊松分布：
Poisson distribution is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval. Let x have a Poisson distribution:

$$P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

其中k是发生次数，参数λ是平均事件数，均值和方差都等于λ：$E[x] = \text{Var}(x) = \lambda$。
where k is occurrence number, parameter λ is average number of events, and mean and variance are both λ: $E[x] = \text{Var}(x) = \lambda$.

给定一组独立同分布样本 $D = \{k^{(1)}, \ldots, k^{(M)}\}$，推导参数λ的最大似然估计。
Given a set of independent and identically distributed (i.i.d.) samples $D = \{k^{(1)}, \ldots, k^{(M)}\}$, derive the maximum-likelihood estimate of λ.

**详细解答 / Detailed Solution:**

**步骤1：写出似然函数 / Step 1: Write Likelihood Function**

$$L(\lambda) = \prod_{i=1}^M P(X=k^{(i)} | \lambda) = \prod_{i=1}^M \frac{\lambda^{k^{(i)}} e^{-\lambda}}{k^{(i)}!}$$

**步骤2：写出对数似然 / Step 2: Write Log-Likelihood**

$$\ell(\lambda) = \log L(\lambda) = \sum_{i=1}^M \left[k^{(i)} \log \lambda - \lambda - \log(k^{(i)}!)\right]$$

**步骤3：对λ求导 / Step 3: Take Derivative w.r.t. λ**

$$\frac{d\ell}{d\lambda} = \sum_{i=1}^M \left[\frac{k^{(i)}}{\lambda} - 1\right] = \frac{1}{\lambda}\sum_{i=1}^M k^{(i)} - M$$

**步骤4：令导数为零 / Step 4: Set Derivative to Zero**

$$\frac{1}{\lambda}\sum_{i=1}^M k^{(i)} - M = 0$$

$$\frac{1}{\lambda}\sum_{i=1}^M k^{(i)} = M$$

**步骤5：求解λ / Step 5: Solve for λ**

$$\lambda^* = \frac{1}{M}\sum_{i=1}^M k^{(i)} = \bar{k}$$

其中 $\bar{k}$ 是样本均值。
where $\bar{k}$ is the sample mean.

**结论 / Conclusion:**
泊松分布参数λ的最大似然估计就是样本的平均值。
MLE of λ for Poisson distribution is simply the sample mean.

---

**题目 / Question (b):**  
下表列出了观察到k次事件的间隔数（可能每分钟）。总间隔数为230。请计算最大似然估计 $\lambda^*$。
The following table lists the number of intervals that are observed to have k occurrences. The total number of intervals is 230. Please calculate the maximum likelihood estimate $\lambda^*$.

| 发生次数 k | 0 | 1 | 2 | 3 | 4+ |
| Occurrences k | 0 | 1 | 2 | 3 | 4+ |
| 间隔数 | 100 | 81 | 34 | 9 | 6 |
| Intervals | 100 | 81 | 34 | 9 | 6 |

**详细解答 / Detailed Solution:**

**步骤1：理解数据 / Step 1: Understand Data**

总间隔数：230
Total intervals: 230

**步骤2：计算总事件数 / Step 2: Calculate Total Events**

假设"4+"表示实际发生4次事件：
Assuming "4+" means actually 4 events occurred:

$$\sum_{i=1}^{230} k^{(i)} = 0 \times 100 + 1 \times 81 + 2 \times 34 + 3 \times 9 + 4 \times 6$$

$$= 0 + 81 + 68 + 27 + 24 = 200$$

**步骤3：计算最大似然估计 / Step 3: Calculate MLE**

$$\lambda^* = \frac{1}{230} \times 200 = \frac{200}{230} = \frac{20}{23} \approx 0.87$$

**结论 / Conclusion:**
最大似然估计 $\lambda^* \approx 0.87$，表示平均每个时间间隔发生约0.87个事件。
MLE $\lambda^* \approx 0.87$, meaning average 0.87 events per time interval.

**关键词 / Keywords:** 对数似然、求导、样本均值、泊松分布。

### 例题2：梯度下降优化非线性误差函数 / Gradient Descent for Nonlinear Error Surface

**题目 / Question:**  
考虑非线性误差表面 $l(u, v) = (ue^v - 2ve^{-u})^2$。我们从点 $(u, v) = (1, 1)$ 开始，使用梯度下降在u, v空间中最小化这个误差。使用学习率 $\alpha = 0.1$。
Consider the nonlinear error surface $l(u, v) = (ue^v - 2ve^{-u})^2$. We start at the point $(u, v) = (1, 1)$ and minimize this error using gradient descent in the u, v space. Use learning rate $\alpha = 0.1$.

a) [1分] l(u, v) 对u的偏导数是什么？
a) [1 point] What is the partial derivative of l(u, v) with respect to u?

b) [1分] 误差 l(u, v) 首次降到 $10^{-14}$ 以下需要多少次迭代？在程序中确保使用双精度以获得所需精度。
b) [1 point] How many iterations does it take for the error l(u, v) to fall below $10^{-14}$ for the first time? In your programs, make sure to use double precision to get the needed accuracy.

c) [1分] 运行足够多次迭代使误差刚好降到 $10^{-14}$ 以下后，问题b)中得到的最终 (u, v) 是什么？将答案四舍五入到千分位。
c) [1 point] After running enough iterations such that the error has just dropped below $10^{-14}$, what is the final (u, v) you get in problem b)? Round your answer to the thousandths place.

**详细解答 / Detailed Solution:**

**部分a)：计算偏导数 / Part a): Calculate Partial Derivative**

**步骤1：识别函数形式 / Step 1: Identify Function Form**

设 $f(u,v) = ue^v - 2ve^{-u}$，则 $l(u,v) = [f(u,v)]^2$
Let $f(u,v) = ue^v - 2ve^{-u}$, then $l(u,v) = [f(u,v)]^2$

**步骤2：应用链式法则 / Step 2: Apply Chain Rule**

$$\frac{\partial l}{\partial u} = 2f(u,v) \cdot \frac{\partial f}{\partial u}$$

**步骤3：计算f对u的偏导数 / Step 3: Calculate Partial Derivative of f w.r.t. u**

$$\frac{\partial f}{\partial u} = \frac{\partial}{\partial u}(ue^v - 2ve^{-u}) = e^v + 2ve^{-u}$$

**步骤4：得到最终结果 / Step 4: Get Final Result**

$$\frac{\partial l}{\partial u} = 2(ue^v - 2ve^{-u})(e^v + 2ve^{-u})$$

类似地，对v的偏导数：
Similarly, partial derivative w.r.t. v:

$$\frac{\partial l}{\partial v} = 2(ue^v - 2ve^{-u})(ue^v - 2e^{-u})$$

---

**部分b)和c)：梯度下降迭代 / Parts b) and c): Gradient Descent Iterations**

**Python参考代码 / Python Reference Code:**

```python
import math

def gradient_descent(fn, partial_derivatives, n_variables, 
                     lr=0.1, max_iter=20, tolerance=1e-14):
    theta = [1, 1]  # 初始值
    y_cur = fn(*theta)
    
    for i in range(max_iter):
        # 计算梯度
        gradient = [df(*theta) for df in partial_derivatives]
        
        # 更新参数
        for j in range(n_variables):
            theta[j] -= gradient[j] * lr
        
        # 检查收敛
        y_cur, y_pre = fn(*theta), y_cur
        if y_cur < tolerance:
            print(i + 1)  # 迭代次数
            break
    
    return theta

def f(u, v):
    return ((u * math.exp(v) - 2 * v * math.exp(-u)) ** 2)

def df_du(u, v):
    return 2 * (u * math.exp(v) - 2 * v * math.exp(-u)) * \
           (math.exp(v) + 2 * v * math.exp(-u))

def df_dv(u, v):
    return 2 * (u * math.exp(v) - 2 * v * math.exp(-u)) * \
           (u * math.exp(v) - 2 * math.exp(-u))

# 运行梯度下降
para = gradient_descent(f, [df_du, df_dv], 2)
para = [round(x, 3) for x in para]
print(f"The solution is: (u, v): {para}")
```

**答案 / Answers:**
- b) **10次迭代** / **10 iterations**
- c) **(0.045, 0.024)**（四舍五入到千分位）
- c) **(0.045, 0.024)** (rounded to thousandths place)

**关键词 / Keywords:** 偏导、链式法则、停止条件、梯度下降、非线性优化。

### 例题3：约束优化与KKT条件 / Constrained Optimization and KKT Conditions

**题目 / Question:**  
考虑以下带不等式约束的二次优化问题：
Consider the following quadratic optimization problem with inequality constraints:

$$\begin{aligned}
\min\; &f(x_1, x_2) = x_1^2 + x_2^2 + 2x_1 + 3x_2 \\
\text{s.t. }& x_1 + x_2 \ge 1 \\
& x_1 - 2x_2 \le 2 \\
& x_1 \ge 0
\end{aligned}$$

这个问题出现在资源分配场景中，我们需要在满足多个线性约束的同时最小化二次成本函数。
This problem arises in resource allocation scenarios where we need to minimize a quadratic cost function while satisfying multiple linear constraints.

a) [1分] 写出这个优化问题的拉格朗日函数和KKT条件。
a) [1 point] Write down the Lagrangian function and the KKT conditions for this optimization problem.

b) [1分] 识别所有满足KKT条件的点。这些点中哪个是全局最优解？
b) [1 point] Identify all points that satisfy the KKT conditions. Which of these points is the global optimum?

c) [1分] 假设第一个约束 $(x_1 + x_2 \ge 1)$ 放宽为 $(x_1 + x_2 \ge 0.5)$。这会对最优解和最优值产生什么影响？使用KKT条件解释。
c) [1 point] Suppose the first constraint $(x_1 + x_2 \ge 1)$ is relaxed to $(x_1 + x_2 \ge 0.5)$. How would this affect the optimal solution and the optimal value? Explain using the KKT conditions.

**详细解答 / Detailed Solution:**

**部分a)：拉格朗日函数和KKT条件 / Part a): Lagrangian and KKT Conditions**

**步骤1：将约束转换为标准形式 / Step 1: Convert Constraints to Standard Form**

标准形式要求约束是"≤0"：
Standard form requires constraints to be "≤0":

- $x_1 + x_2 \ge 1$ → $1 - x_1 - x_2 \le 0$ (记作 $g_1(x) \le 0$)
- $x_1 - 2x_2 \le 2$ → $x_1 - 2x_2 - 2 \le 0$ (记作 $g_2(x) \le 0$)
- $x_1 \ge 0$ → $-x_1 \le 0$ (记作 $g_3(x) \le 0$)

**步骤2：写出拉格朗日函数 / Step 2: Write Lagrangian Function**

$$L(x_1, x_2, \lambda_1, \lambda_2, \lambda_3) = x_1^2 + x_2^2 + 2x_1 + 3x_2 + \lambda_1(1-x_1-x_2) + \lambda_2(x_1-2x_2-2) + \lambda_3(-x_1)$$

**步骤3：写出KKT条件 / Step 3: Write KKT Conditions**

**平稳性条件 / Stationarity:**
$$\frac{\partial L}{\partial x_1} = 2x_1 + 2 - \lambda_1 + \lambda_2 - \lambda_3 = 0$$
$$\frac{\partial L}{\partial x_2} = 2x_2 + 3 - \lambda_1 - 2\lambda_2 = 0$$

**原始可行性 / Primal Feasibility:**
$$1 - x_1 - x_2 \le 0, \quad x_1 - 2x_2 - 2 \le 0, \quad -x_1 \le 0$$

**对偶可行性 / Dual Feasibility:**
$$\lambda_1 \ge 0, \quad \lambda_2 \ge 0, \quad \lambda_3 \ge 0$$

**互补松弛性 / Complementary Slackness:**
$$\lambda_1(1 - x_1 - x_2) = 0$$
$$\lambda_2(x_1 - 2x_2 - 2) = 0$$
$$\lambda_3(-x_1) = 0$$

---

**部分b)：寻找KKT点 / Part b): Finding KKT Points**

**步骤1：检查无约束最小值 / Step 1: Check Unconstrained Minimum**

无约束最小值：
Unconstrained minimum:

$$\nabla f = 0 \Rightarrow 2x_1 + 2 = 0, \quad 2x_2 + 3 = 0$$

$$x_1 = -2, \quad x_2 = -1.5$$

**检查可行性 / Check Feasibility:**
- $x_1 = -2 < 0$ ✗（违反 $x_1 \ge 0$）
- $x_1 = -2 < 0$ ✗ (violates $x_1 \ge 0$)
- $x_1 + x_2 = -3.5 < 1$ ✗（违反 $x_1 + x_2 \ge 1$）
- $x_1 + x_2 = -3.5 < 1$ ✗ (violates $x_1 + x_2 \ge 1$)

**结论：无约束最小值不可行，必须激活某些约束。**
**Conclusion: Unconstrained minimum is infeasible, some constraints must be active.**

**步骤2：尝试激活第一个约束 / Step 2: Try Activating First Constraint**

假设第一个约束激活：$x_1 + x_2 = 1$，其他约束不激活：$\lambda_2 = \lambda_3 = 0$
Assume first constraint active: $x_1 + x_2 = 1$, others inactive: $\lambda_2 = \lambda_3 = 0$

从平稳性条件：
From stationarity conditions:

$$2x_1 + 2 - \lambda_1 = 0 \Rightarrow \lambda_1 = 2x_1 + 2$$
$$2x_2 + 3 - \lambda_1 = 0 \Rightarrow \lambda_1 = 2x_2 + 3$$

结合约束 $x_1 + x_2 = 1$：
Combining with constraint $x_1 + x_2 = 1$:

$$2x_1 + 2 = 2x_2 + 3 \Rightarrow 2x_1 - 2x_2 = 1$$

与 $x_1 + x_2 = 1$ 联立求解：
Solving together with $x_1 + x_2 = 1$:

$$x_1 = 1, \quad x_2 = 0, \quad \lambda_1 = 4$$

**验证可行性 / Verify Feasibility:**
- $x_1 + x_2 = 1 \ge 1$ ✓
- $x_1 - 2x_2 = 1 \le 2$ ✓
- $x_1 = 1 \ge 0$ ✓
- $\lambda_1 = 4 > 0$ ✓

**计算目标值 / Calculate Objective Value:**
$$f(1, 0) = 1^2 + 0^2 + 2(1) + 3(0) = 1 + 0 + 2 + 0 = 2.5$$

**结论 / Conclusion:**
唯一的KKT点（也是全局最优解）是：
Unique KKT point (and global optimum) is:

$$x^* = (1, 0), \quad f^* = 2.5$$

---

**部分c)：放宽约束的影响 / Part c): Effect of Relaxing Constraint**

**步骤1：新约束 / Step 1: New Constraint**

新约束：$x_1 + x_2 \ge 0.5$，即 $0.5 - x_1 - x_2 \le 0$
New constraint: $x_1 + x_2 \ge 0.5$, i.e., $0.5 - x_1 - x_2 \le 0$

**步骤2：假设只有新约束激活 / Step 2: Assume Only New Constraint Active**

假设 $x_1 + x_2 = 0.5$，$\lambda_2 = \lambda_3 = 0$
Assume $x_1 + x_2 = 0.5$, $\lambda_2 = \lambda_3 = 0$

类似求解得到：
Similar solving gives:

$$x_1 = \frac{1}{3}, \quad x_2 = \frac{1}{6}, \quad \lambda_1 = \frac{8}{3}$$

**验证其他约束 / Verify Other Constraints:**
- $x_1 - 2x_2 = \frac{1}{3} - \frac{2}{6} = 0 \le 2$ ✓
- $x_1 = \frac{1}{3} \ge 0$ ✓

**计算新目标值 / Calculate New Objective Value:**
$$f\left(\frac{1}{3}, \frac{1}{6}\right) = \left(\frac{1}{3}\right)^2 + \left(\frac{1}{6}\right)^2 + 2\left(\frac{1}{3}\right) + 3\left(\frac{1}{6}\right) = \frac{13}{12}$$

**结论 / Conclusion:**
- 原最优值：$f^* = 2.5$
- Original optimal value: $f^* = 2.5$
- 新最优值：$f^* = \frac{13}{12} \approx 1.083$
- New optimal value: $f^* = \frac{13}{12} \approx 1.083$
- **最优值减小了**（因为可行域扩大了）
- **Optimal value decreased** (because feasible region expanded)

**关键词 / Keywords:** 互补松弛、活跃约束、可行性、KKT条件、约束优化。

### 例题4：点到超平面的距离 / Distance from Point to Hyperplane

**题目 / Question:**  
在SVM的公式中，我们需要计算N维空间中任意点 $x^{(i)}$ 到超平面 $w^\top x + b = 0$ 的距离（即间隔），这可以表述为以下优化问题：
In the formulation of SVM, we need to compute the margin (i.e., the distance) between an arbitrary point $x^{(i)}$ in the N-dimensional space and a hyperplane $w^\top x + b = 0$, which can be formulated as the following optimization problem:

$$\min_x \|x^{(i)} - x\|^2 \quad \text{s.t. } w^\top x + b = 0$$

a) [1分] 这个问题是凸的吗？为什么？
a) [1 point] Is this problem convex and why?

b) [2分] 使用拉格朗日对偶性求解最优x和距离。（记住要形成拉格朗日函数并推导拉格朗日对偶函数）。
b) [2 points] Using the Lagrange duality to solve for the optimal x and the distance. (Remember to form the Lagrangian and derive the Lagrange dual function).

**详细解答 / Detailed Solution:**

**部分a)：凸性分析 / Part a): Convexity Analysis**

**答案：是的，这个问题是凸的。**
**Answer: Yes, this problem is convex.**

**原因 / Reason:**
1. **目标函数是凸的 / Objective function is convex:**
   - $\|x^{(i)} - x\|^2$ 是仿射函数的L2范数的平方
   - $\|x^{(i)} - x\|^2$ is the square of L2 norm of an affine function
   - 仿射函数的L2范数是凸函数
   - L2 norm of affine function is convex
   - 凸函数的平方仍然是凸函数
   - Square of convex function is still convex

2. **约束是仿射的 / Constraint is affine:**
   - $w^\top x + b = 0$ 是线性等式约束
   - $w^\top x + b = 0$ is a linear equality constraint
   - 仿射约束保持凸性
   - Affine constraints preserve convexity

**结论：这是一个凸优化问题。**
**Conclusion: This is a convex optimization problem.**

---

**部分b)：使用拉格朗日对偶求解 / Part b): Solve Using Lagrange Duality**

**步骤1：等价问题 / Step 1: Equivalent Problem**

给定的凸优化问题等价于：
Given convex optimization problem is equivalent to:

$$\min_x \frac{1}{2}\|x^{(i)} - x\|^2 \quad \text{s.t. } w^\top x + b = 0$$

（乘以1/2不改变最优解）
(Multiplying by 1/2 doesn't change optimal solution)

**步骤2：形成拉格朗日函数 / Step 2: Form Lagrangian**

$$L(x, \lambda) = \frac{1}{2}\|x^{(i)} - x\|^2 + \lambda(w^\top x + b)$$

其中λ是拉格朗日乘数。
where λ is Lagrange multiplier.

**步骤3：对x求导并令其为零 / Step 3: Take Derivative w.r.t. x and Set to Zero**

$$\frac{\partial L}{\partial x} = -(x^{(i)} - x) + \lambda w = 0$$

$$x^{(i)} - x = \lambda w$$

$$x = x^{(i)} - \lambda w \quad \text{(10)}$$

**步骤4：代入约束求解λ / Step 4: Substitute into Constraint to Solve for λ**

将式(10)代入约束：
Substitute equation (10) into constraint:

$$w^\top(x^{(i)} - \lambda w) + b = 0$$

$$w^\top x^{(i)} - \lambda w^\top w + b = 0$$

$$\lambda = \frac{w^\top x^{(i)} + b}{w^\top w}$$

**步骤5：推导拉格朗日对偶函数 / Step 5: Derive Lagrange Dual Function**

将式(10)代回拉格朗日函数：
Substitute equation (10) back into Lagrangian:

$$g(\lambda) = \frac{1}{2}\|\lambda w\|^2 + \lambda(w^\top(x^{(i)} - \lambda w) + b)$$

$$= \frac{1}{2}\lambda^2 w^\top w + \lambda(w^\top x^{(i)} + b) - \lambda^2 w^\top w$$

$$= \lambda(w^\top x^{(i)} + b) - \frac{1}{2}\lambda^2 w^\top w$$

**步骤6：最大化对偶函数 / Step 6: Maximize Dual Function**

对λ求导并令其为零：
Take derivative w.r.t. λ and set to zero:

$$\frac{dg}{d\lambda} = w^\top x^{(i)} + b - \lambda w^\top w = 0$$

$$\lambda^* = \frac{w^\top x^{(i)} + b}{w^\top w}$$

**步骤7：计算距离 / Step 7: Calculate Distance**

最优解：
Optimal solution:

$$x^* = x^{(i)} - \lambda^* w = x^{(i)} - \frac{w^\top x^{(i)} + b}{w^\top w} w$$

距离：
Distance:

$$d = \|x^{(i)} - x^*\| = \|\lambda^* w\| = |\lambda^*| \|w\| = \frac{|w^\top x^{(i)} + b|}{\|w\|}$$

**结论 / Conclusion:**
点到超平面的距离为：
Distance from point to hyperplane is:

$$d = \frac{|w^\top x^{(i)} + b|}{\|w\|}$$

这正是我们想要的结果。
This is exactly the desired result.

---

### 例题5：硬间隔SVM的对偶形式 / Dual Form of Hard Margin SVM

**题目 / Question:**  
在课程笔记中，我们详细推导了软间隔SVM的对偶形式。使用更简单的论证，推导硬间隔SVM的对偶形式：
In the lecture note, we have given a detailed derivation of the dual form of SVM with soft margin. With simpler arguments, derive the dual form of SVM with hard margin:

$$\begin{aligned}
\min_{w,b}\; & \frac{1}{2}w^\top w \\
\text{s.t. }& y^{(i)}(w^\top x^{(i)} + b) \ge 1, \quad i = 1, \ldots, M
\end{aligned}$$

比较两种对偶形式。
Compare the two dual forms.

**详细解答 / Detailed Solution:**

**步骤1：形成拉格朗日函数 / Step 1: Form Lagrangian**

$$L(w, b, \alpha) = \frac{1}{2}w^\top w - \sum_{i=1}^M \alpha_i[y^{(i)}(w^\top x^{(i)} + b) - 1]$$

其中 $\alpha_i \ge 0$ 是拉格朗日乘数。
where $\alpha_i \ge 0$ are Lagrange multipliers.

**步骤2：对w和b求偏导并令其为零 / Step 2: Take Partial Derivatives w.r.t. w and b and Set to Zero**

$$\frac{\partial L}{\partial w} = w - \sum_{i=1}^M \alpha_i y^{(i)} x^{(i)} = 0 \quad \Rightarrow \quad w = \sum_{i=1}^M \alpha_i y^{(i)} x^{(i)} \quad \text{(11)}$$

$$\frac{\partial L}{\partial b} = -\sum_{i=1}^M \alpha_i y^{(i)} = 0 \quad \Rightarrow \quad \sum_{i=1}^M \alpha_i y^{(i)} = 0 \quad \text{(12)}$$

**步骤3：将式(11)和(12)代回拉格朗日函数 / Step 3: Substitute Equations (11) and (12) into Lagrangian**

$$L(w, b, \alpha) = \frac{1}{2}\left(\sum_{i=1}^M \alpha_i y^{(i)} x^{(i)}\right)^\top \left(\sum_{j=1}^M \alpha_j y^{(j)} x^{(j)}\right) - \sum_{i=1}^M \alpha_i\left[y^{(i)}\left(\sum_{j=1}^M \alpha_j y^{(j)} (x^{(j)})^\top x^{(i)} + b\right) - 1\right]$$

简化后：
After simplification:

$$g(\alpha) = \sum_{i=1}^M \alpha_i - \frac{1}{2}\sum_{i=1}^M \sum_{j=1}^M \alpha_i \alpha_j y^{(i)} y^{(j)} (x^{(i)})^\top x^{(j)}$$

**步骤4：写出对偶问题 / Step 4: Write Dual Problem**

$$\begin{aligned}
\max_{\alpha}\;& \sum_{i=1}^M \alpha_i - \frac{1}{2}\sum_{i=1}^M \sum_{j=1}^M \alpha_i \alpha_j y^{(i)} y^{(j)} (x^{(i)})^\top x^{(j)} \\
\text{s.t. }& \sum_{i=1}^M \alpha_i y^{(i)} = 0 \\
& \alpha_i \ge 0, \quad i = 1, \ldots, M
\end{aligned}$$

**步骤5：与软间隔SVM比较 / Step 5: Compare with Soft Margin SVM**

**硬间隔SVM对偶 / Hard Margin SVM Dual:**
$$\alpha_i \ge 0, \quad i = 1, \ldots, M$$

**软间隔SVM对偶 / Soft Margin SVM Dual:**
$$0 \le \alpha_i \le C, \quad i = 1, \ldots, M$$

**关键区别 / Key Difference:**
- **硬间隔**：$\alpha_i$ 只有下界0，没有上界
- **Hard margin**: $\alpha_i$ only has lower bound 0, no upper bound
- **软间隔**：$\alpha_i$ 有上界C，C控制间隔和松弛的权衡
- **Soft margin**: $\alpha_i$ has upper bound C, C controls trade-off between margin and slack

**关键词 / Keywords:** 拉格朗日对偶、惩罚系数C、支持向量约束、硬间隔、软间隔。

