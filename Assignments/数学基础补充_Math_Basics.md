# 数学基础补充 / Math Foundations Supplement

## 目录 / Table of Contents
1. [概率与信息论 / Probability & Information](#prob)
2. [线性代数 / Linear Algebra](#linalg)
3. [微积分与优化 / Calculus & Optimization](#calculus)
4. [常用不等式与技巧 / Useful Inequalities & Tricks](#ineq)
5. [例题与解答 / Worked Examples](#worked-examples)

---

## 概率与信息论 / Probability & Information <a name="prob"></a>

- 条件概率 / Conditional probability: $P(A\mid B) = \dfrac{P(A,B)}{P(B)}$  
- 全概率公式 / Law of total probability: $P(A)=\sum_i P(A\mid B_i)P(B_i)$  
- 贝叶斯公式 / Bayes' rule: $P(A\mid B)=\dfrac{P(B\mid A)P(A)}{P(B)}$  
- 期望 / Expectation: $E[X]=\sum_x x\,P(x)$；$E[aX+b]=aE[X]+b$  
- 方差 / Variance: $\mathrm{Var}(X)=E[X^2]-(E[X])^2$；$\mathrm{Var}(aX+b)=a^2\mathrm{Var}(X)$  
- 熵 / Entropy: $H(X)=-\sum p\log p$；联合、条件、互信息同 Assignment 1。

---

## 线性代数 / Linear Algebra <a name="linalg"></a>

- 矩阵秩 / Rank: 列空间维度；秩决定线性方程可解性。  
- 零空间 / Nullspace: $\{x \mid Ax=0\}$；维度 $= N-\mathrm{rank}(A)$。  
- 正交与正交归一 / Orthogonal & Orthonormal: $Q^\top Q = I$。  
- 特征分解 / Eigen-decomposition: 对称矩阵 $A = Q\Lambda Q^\top$，$\Lambda$ 为实特征值。  
- 迹与行列式 / Trace & Determinant: $\mathrm{tr}(AB)=\mathrm{tr}(BA)$；$\det(AB)=\det(A)\det(B)$。  
- 投影 / Projection: $P = A(A^\top A)^{-1}A^\top$ 将向量投影到列空间。

---

## 微积分与优化 / Calculus & Optimization <a name="calculus"></a>

- 链式法则 / Chain rule: $\dfrac{d}{dx} f(g(x)) = f'(g(x)) g'(x)$。  
- 梯度 / Gradient: $\nabla f = [\partial f/\partial x_1,\dots,\partial f/\partial x_n]$；指向上升最快方向。  
- Hessian: 二阶导矩阵；正定 ⇒ 严格凸。  
- 凸函数 / Convex: $f(\theta x+(1-\theta)y) \le \theta f(x)+(1-\theta)f(y)$。  
- 拉格朗日乘子 / Lagrange multipliers: 处理等式约束。  
- KKT 条件 / KKT: 平稳性、原始可行、对偶可行、互补松弛。

---

## 常用不等式与技巧 / Useful Inequalities & Tricks <a name="ineq"></a>

- Jensen 不等式: 对凸函数 $f$，$f(E[X]) \le E[f(X)]$。  
- Cauchy–Schwarz: $|a^\top b| \le \|a\|\,\|b\|$。  
- AM-GM: $\dfrac{a+b}{2} \ge \sqrt{ab},\; a,b\ge0$。  
- 对数恒等: $\log ab = \log a + \log b$；$\log \frac{a}{b} = \log a - \log b$。  
- 迹技巧: $\mathrm{tr}(AB)=\mathrm{tr}(BA)$，便于化简二次型。  
- 指数与对数导数: $\dfrac{d}{dx}\ln f = \dfrac{f'}{f}$。

---

## 例题与解答 / Worked Examples <a name="worked-examples"></a>

### 例题1：条件概率计算 / Conditional Probability
**题目 / Question:** P(A)=0.4, P(B)=0.5, P(A,B)=0.2，求 P(A|B)。  
**解答 / Solution (中):** P(A|B)=P(A,B)/P(B)=0.2/0.5=0.4。  
**Solution (En):** P(A|B) = 0.2 / 0.5 = 0.4.  
**关键词:** 贝叶斯、条件概率。

### 例题2：方差平移与缩放 / Variance Shift & Scale
**题目 / Question:** Y = 3X − 2，已知 Var(X)=5，求 Var(Y)。  
**解答 / Solution (中):** Var(Y)=3² Var(X)=9×5=45。  
**Solution (En):** Var(Y) = 9 × 5 = 45.  
**关键词:** 方差缩放、线性变换。

### 例题3：Cauchy–Schwarz 应用 / Applying Cauchy–Schwarz
**题目 / Question:** 证明 |xᵀy| ≤ ||x||₂ ||y||₂。  
**解答 / Solution（中要点）:**  
1) ||x−ty||² ≥ 0 对任意 t；展开得 t²||y||² − 2t xᵀy + ||x||² ≥ 0。  
2) 判别式 ≤ 0 ⇒ (xᵀy)² ≤ ||x||²||y||²。取平方根得结论。  
**Solution (En brief):** Nonnegativity of ||x−ty||² gives discriminant ≤ 0 ⇒ (xᵀy)² ≤ ||x||²||y||² ⇒ |xᵀy| ≤ ||x||·||y||.  
**关键词:** 判别式、非负性。

### 例题4：凸性判定 / Convexity Check
**题目 / Question:** f(x)=x² 是否凸？  
**解答 / Solution (中):** f''(x)=2>0，凸；也可用 Jensen 证明。  
**Solution (En):** Second derivative 2>0 ⇒ convex; Jensen also applies.  
**关键词:** 二阶导、Jensen。

