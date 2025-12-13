# Assignment 3 学习笔记 / Study Notes

## 目录 / Table of Contents
1. [矩阵恒等式与Woodbury公式 / Matrix Identities & Woodbury](#woodbury)
2. [线性方程组与秩 / Linear Systems & Rank](#linear-systems)
3. [坐标下降与线性回归 / Coordinate Descent for Linear Regression](#coordinate-descent)
4. [软间隔SVM的L2松弛 / L2-Slack Soft-Margin SVM](#l2-svm)
5. [例题与解答 / Worked Examples](#worked-examples)

---

## 矩阵恒等式与Woodbury公式 / Matrix Identities & Woodbury <a name="woodbury"></a>

**核心恒等式 / Key Identity (Woodbury):**  
$$
(P^{-1} + B^\top R^{-1}B)^{-1} B^\top R^{-1} = P B^\top (B P B^\top + R)^{-1}
$$

**符号说明 / Symbol Explanation:**
- $P$：n×n的可逆矩阵 / n×n invertible matrix
- $B$：m×n的矩阵 / m×n matrix
- $R$：m×m的可逆矩阵 / m×m invertible matrix
- $P^{-1}$：矩阵P的逆矩阵 / Inverse matrix of P
- $R^{-1}$：矩阵R的逆矩阵 / Inverse matrix of R
- $B^\top$：矩阵B的转置 / Transpose of matrix B
- $(P^{-1} + B^\top R^{-1}B)^{-1}$：左侧需要计算n×n矩阵的逆 / Left side: need to compute inverse of n×n matrix
- $(B P B^\top + R)^{-1}$：右侧需要计算m×m矩阵的逆 / Right side: need to compute inverse of m×m matrix

**适用场景 / When to Use:**
- 当 m ≪ n（m远小于n）时，右侧更高效 / When m ≪ n, right side is more efficient
- 因为计算m×m矩阵的逆比计算n×n矩阵的逆快得多 / Because computing inverse of m×m matrix is much faster than n×n matrix

**计算步骤（使用右侧公式）/ Calculation Steps (Using Right Side):**
1. 计算 $B P B^\top$（矩阵乘法，结果大小为m×m）/ Calculate $B P B^\top$ (matrix multiplication, result size m×m)
2. 计算 $B P B^\top + R$（矩阵加法）/ Calculate $B P B^\top + R$ (matrix addition)
3. 计算 $(B P B^\top + R)^{-1}$（求m×m矩阵的逆）/ Calculate $(B P B^\top + R)^{-1}$ (inverse of m×m matrix)
4. 计算 $P B^\top$（矩阵乘法，结果大小为n×m）/ Calculate $P B^\top$ (matrix multiplication, result size n×m)
5. 计算 $P B^\top (B P B^\top + R)^{-1}$（最终矩阵乘法）/ Calculate $P B^\top (B P B^\top + R)^{-1}$ (final matrix multiplication)

**计算示例 / Calculation Example:**
假设P是3×3矩阵，B是2×3矩阵，R是2×2矩阵（m=2, n=3，所以m<n）
Suppose P is 3×3, B is 2×3, R is 2×2 (m=2, n=3, so m<n)

具体数值：
P = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
B = [[1, 1, 0], [0, 1, 1]]
R = [[1, 0], [0, 1]]

使用右侧公式（更高效，因为只需计算2×2矩阵的逆）:
Using right side formula (more efficient, only need to compute inverse of 2×2 matrix):

步骤1: BPB^T = [[1,1,0],[0,1,1]] × [[1,0,0],[0,2,0],[0,0,3]] × [[1,0],[1,1],[0,1]]
        = [[1,2,0],[0,2,3]] × [[1,0],[1,1],[0,1]] = [[3,2],[2,5]]

步骤2: BPB^T + R = [[3,2],[2,5]] + [[1,0],[0,1]] = [[4,2],[2,6]]

步骤3: (BPB^T + R)^(-1) = [[4,2],[2,6]]^(-1) = [[0.3, -0.1], [-0.1, 0.2]]

步骤4: PB^T = [[1,0,0],[0,2,0],[0,0,3]] × [[1,0],[1,1],[0,1]] = [[1,0],[2,2],[0,3]]

步骤5: PB^T (BPB^T + R)^(-1) = [[1,0],[2,2],[0,3]] × [[0.3,-0.1],[-0.1,0.2]]
        = [[0.3, -0.1], [0.4, 0.2], [-0.3, 0.6]]

如果使用左侧公式，需要计算3×3矩阵的逆，计算量更大
If using left side formula, need to compute inverse of 3×3 matrix, more computation

**证明要点 / Proof Sketch:**
- 右乘 (BPBᵀ + R)，化简到 P Bᵀ。
- 利用分块逆与乘法结合律。

---

## 线性方程组与秩 / Linear Systems & Rank <a name="linear-systems"></a>

**问题形式 / Form:** Ax = y, A ∈ ℝ^{M×N}.

**关键结论 / Key Results:**
- 若 N = M，不一定唯一解（需 det(A) ≠ 0）。
- 若 N > M，未必可解，但零空间维度 Nullity(A) ≥ N−M > 0。
- Rank-nullity: rank(A) + nullity(A) = N。
- 若 N < M，可能无解（过定约束）。

**直观 / Intuition:** 列空间决定可到达的 y；零空间决定解的自由度。

---

## 坐标下降与线性回归 / Coordinate Descent for Linear Regression <a name="coordinate-descent"></a>

**目标 / Objective:** 最小化  
$$
\tfrac{1}{2}\sum_i \big(y^{(i)} - w^\top x^{(i)}\big)^2
$$

**更新式 / Update Rule (选择第 k 维):**  
$$
w_k \leftarrow \frac{\sum_i x_k^{(i)}\big(y^{(i)} - r^{(i)} + w_k x_k^{(i)}\big)}{\sum_i (x_k^{(i)})^2}
$$

**计算示例 / Calculation Example:**
假设有3个样本，当前权重w = [2, 3]，要更新w₁（k=1）
Suppose 3 samples, current weights w = [2, 3], want to update w₁ (k=1)

样本数据：
x^(1) = [1, 2], y^(1) = 5
x^(2) = [2, 1], y^(2) = 7
x^(3) = [1, 1], y^(3) = 4

计算当前残差 / Calculate current residuals:
- r^(1) = 5 - (2×1 + 3×2) = 5 - 8 = -3
- r^(2) = 7 - (2×2 + 3×1) = 7 - 7 = 0
- r^(3) = 4 - (2×1 + 3×1) = 4 - 5 = -1

更新w₁ / Update w₁:
- 分子: x₁^(1)(y^(1) - r^(1) + w₁x₁^(1)) + x₁^(2)(y^(2) - r^(2) + w₁x₁^(2)) + x₁^(3)(y^(3) - r^(3) + w₁x₁^(3))
- = 1×(5 - (-3) + 2×1) + 2×(7 - 0 + 2×2) + 1×(4 - (-1) + 2×1)
- = 1×(8 + 2) + 2×(7 + 4) + 1×(5 + 2)
- = 10 + 22 + 7 = 39

- 分母: (x₁^(1))² + (x₁^(2))² + (x₁^(3))² = 1² + 2² + 1² = 1 + 4 + 1 = 6

- 新w₁ = 39 / 6 = 6.5

**符号说明 / Symbol Explanation:**
- $w_k$：第k个特征的权重系数 / Weight coefficient for k-th feature
- $\leftarrow$：赋值符号，表示更新 / Assignment symbol, means update
- $\sum_i$：对所有训练样本i求和 / Sum over all training samples i
- $x_k^{(i)}$：第i个样本的第k个特征值 / k-th feature value of i-th sample
- $y^{(i)}$：第i个样本的真实标签值 / True label value of i-th sample
- $r^{(i)}$：第i个样本的残差，$r^{(i)} = y^{(i)} - \sum_{j} w_j x_j^{(i)}$ / Residual of i-th sample, $r^{(i)} = y^{(i)} - \sum_{j} w_j x_j^{(i)}$
- $w_k x_k^{(i)}$：当前权重$w_k$对第i个样本的贡献 / Contribution of current weight $w_k$ to i-th sample
- $(x_k^{(i)})^2$：第k个特征值的平方 / Square of k-th feature value

**计算步骤 / Calculation Steps:**
1. 计算当前残差：$r^{(i)} = y^{(i)} - \sum_{j} w_j x_j^{(i)}$（对所有样本i）/ Calculate current residuals: $r^{(i)} = y^{(i)} - \sum_{j} w_j x_j^{(i)}$ (for all samples i)
2. 对每个样本i，计算 $x_k^{(i)} \times (y^{(i)} - r^{(i)} + w_k x_k^{(i)})$ / For each sample i, calculate $x_k^{(i)} \times (y^{(i)} - r^{(i)} + w_k x_k^{(i)})$
3. 将所有样本的结果相加得到分子 / Sum results from all samples to get numerator
4. 计算分母：$\sum_i (x_k^{(i)})^2$（所有样本第k个特征值的平方和）/ Calculate denominator: $\sum_i (x_k^{(i)})^2$ (sum of squares of k-th feature)
5. 用分子除以分母得到新的$w_k$ / Divide numerator by denominator to get new $w_k$

**等价增量式 / Equivalent Incremental Update:**  
$$
\Delta = \frac{\sum_i x_k^{(i)} r^{(i)}}{\sum_i (x_k^{(i)})^2},\quad
w_k \leftarrow w_k + \Delta,\quad
r^{(i)} \leftarrow r^{(i)} - \Delta x_k^{(i)}
$$
增量式更高效：O(MN) 而非 O(MN²)。

---

## 软间隔SVM的L2松弛 / L2-Slack Soft-Margin SVM <a name="l2-svm"></a>

**原始形式 / Primal:**  
$$
\min_{w,b,\xi}\; \tfrac{1}{2}\|w\|^2 + C\sum_i \xi_i^2 \quad
\text{s.t. } y_i(w^\top \phi(x_i)+b) \ge 1 - \xi_i
$$
非负约束 ξ_i ≥ 0 可证明冗余：若 ξ_i < 0 且满足间隔，则设为 0 令目标更小。

**对偶要点 / Dual Highlights:**
- 拉格朗日对偶后，变量 α_i 无上界 (与 L1 松弛 0 ≤ α_i ≤ C 不同)。
- 核技巧直接作用于 Gram 矩阵 K(x_i, x_j)。

---

## 例题与解答 / Worked Examples <a name="worked-examples"></a>

### 例题1：Woodbury矩阵恒等式 / Woodbury Matrix Identity

**题目 / Question:**  
证明以下矩阵恒等式：
Prove the following matrix identity:

$$(P^{-1} + B^\top R^{-1} B)^{-1} B^\top R^{-1} = P B^\top (B P B^\top + R)^{-1}$$

其中 $P \in \mathbb{R}^{n \times n}$, $B \in \mathbb{R}^{m \times n}$, $R \in \mathbb{R}^{m \times m}$。P和R可逆。注意如果 $m < n$，计算右边比左边便宜得多。
where $P \in \mathbb{R}^{n \times n}$, $B \in \mathbb{R}^{m \times n}$, $R \in \mathbb{R}^{m \times m}$. P and R are invertible. Note that if $m < n$, it will be much cheaper to evaluate the right-hand side than the left-hand side.

提示：右乘两边 $(B P B^\top + R)$。使用类似论证，证明式(1)的特殊情况：
Hint: Right multiply both sides by $(B P B^\top + R)$. With similar arguments, prove a special case of Eq. (1):

$$(I + AB)^{-1} A = A(I + BA)^{-1}$$

其中 $A \in \mathbb{R}^{n \times m}$, $B \in \mathbb{R}^{m \times n}$。
where $A \in \mathbb{R}^{n \times m}$, $B \in \mathbb{R}^{m \times n}$.

**详细解答 / Detailed Solution:**

**步骤1：证明主要恒等式 / Step 1: Prove Main Identity**

**策略：右乘两边 $(B P B^\top + R)$**
**Strategy: Right multiply both sides by $(B P B^\top + R)$**

**左边 / Left-hand side:**
$$(P^{-1} + B^\top R^{-1} B)^{-1} B^\top R^{-1} (B P B^\top + R)$$

展开：
Expand:

$$= (P^{-1} + B^\top R^{-1} B)^{-1} (B^\top R^{-1} B P B^\top + B^\top R^{-1} R)$$

$$= (P^{-1} + B^\top R^{-1} B)^{-1} (B^\top R^{-1} B P B^\top + B^\top)$$

注意到 $B^\top = P^{-1} P B^\top$：
Note that $B^\top = P^{-1} P B^\top$:

$$= (P^{-1} + B^\top R^{-1} B)^{-1} (B^\top R^{-1} B + P^{-1}) P B^\top$$

$$= (P^{-1} + B^\top R^{-1} B)^{-1} (P^{-1} + B^\top R^{-1} B) P B^\top$$

$$= I \cdot P B^\top = P B^\top$$

**右边 / Right-hand side:**
$$P B^\top (B P B^\top + R)^{-1} (B P B^\top + R) = P B^\top$$

**结论：两边相等，恒等式成立。**
**Conclusion: Both sides equal, identity holds.**

---

**步骤2：证明特殊情况 / Step 2: Prove Special Case**

对于特殊情况 $(I + AB)^{-1} A = A(I + BA)^{-1}$：
For special case $(I + AB)^{-1} A = A(I + BA)^{-1}$:

**方法1：使用主要恒等式 / Method 1: Use Main Identity**

设 $A = P B^\top R^{-1}$，$B = B$（如上面所示）：
Let $A = P B^\top R^{-1}$, $B = B$ (as illustrated above):

从主要恒等式：
From main identity:

$$(P^{-1} + B^\top R^{-1} B)^{-1} B^\top R^{-1} = P B^\top (B P B^\top + R)^{-1}$$

重新排列：
Rearranging:

$$(P^{-1} + B^\top R^{-1} B)^{-1} B^\top R^{-1} = P B^\top R^{-1} (R^{-1} B P B^\top R^{-1} + I)^{-1}$$

设 $A = P B^\top R^{-1}$，则：
Let $A = P B^\top R^{-1}$, then:

$$(I + BA)^{-1} A = A(I + AB)^{-1}$$

**方法2：直接验证 / Method 2: Direct Verification**

右乘 $(I + BA)$：
Right multiply by $(I + BA)$:

$$(I + AB)^{-1} A (I + BA) = (I + AB)^{-1} (A + ABA) = (I + AB)^{-1} (I + AB) A = A$$

因此：
Therefore:

$$(I + AB)^{-1} A = A(I + BA)^{-1} \quad \square$$

**关键词 / Keywords:** 右乘、矩阵逆、交换顺序、Woodbury恒等式。

### 例题2：线性方程组解的存在性 / Existence of Solutions for Linear Systems

**题目 / Question:**  
假设你有M个线性方程，N个变量。矩阵形式写作 $Ax = y$，其中 $A \in \mathbb{R}^{M \times N}$, $x \in \mathbb{R}^{N \times 1}$, $y \in \mathbb{R}^{M \times 1}$。
Say you have M linear equations in N variables. In matrix form we write $Ax = y$, where $A \in \mathbb{R}^{M \times N}$, $x \in \mathbb{R}^{N \times 1}$, and $y \in \mathbb{R}^{M \times 1}$.

对以下每个陈述，给出证明或反例：
Given a proof or a counterexample for each of the following:

a) [1分] 如果 N = M，总是至多有一个解。
a) [1 point] If N = M, there is always at most one solution.

b) [1分] 如果 N > M，你总是可以解 $Ax = y$。
b) [1 point] If N > M, you can always solve $Ax = y$.

c) [1分] 如果 N > M，A的零空间维度大于零。
c) [1 point] If N > M, the nullspace of A has dimension greater than zero.

d) [1分] 如果 N < M，那么对于某些y，$Ax = y$ 无解。
d) [1 point] If N < M, then for some y there is no solution of $Ax = y$.

e) [1分] 如果 N < M，$Ax = 0$ 的唯一解是 $x = 0$。
e) [1 point] If N < M, the only solution of $Ax = 0$ is $x = 0$.

提示：A的零空间V包含满足 $\{x \in V | Ax = 0\}$ 的向量集合。
Hint: The null space of A, denoted by V, contains the set of vectors that satisfy $\{x \in V | Ax = 0\}$.

**详细解答 / Detailed Solution:**

**部分a)：如果 N = M，总是至多有一个解 / Part a): If N = M, Always At Most One Solution**

**答案：错误 / Answer: False**

**反例 / Counterexample:**
考虑：
Consider:

$$A = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}, \quad y = \begin{bmatrix} 2 \\ 2 \end{bmatrix}$$

这里 N = M = 2，但有无穷多个解：
Here N = M = 2, but there are infinitely many solutions:

$$x = \begin{bmatrix} t \\ 2-t \end{bmatrix}, \quad \forall t \in \mathbb{R}$$

**正确陈述：如果 N = M 且 A 可逆，则存在唯一解。**
**Correct statement: If N = M and A is invertible, then there is a unique solution.**

---

**部分b)：如果 N > M，总是可以解 / Part b): If N > M, Always Solvable**

**答案：错误 / Answer: False**

**反例 / Counterexample:**
考虑：
Consider:

$$A = \begin{bmatrix} 1 & 0 \\ 1 & 0 \end{bmatrix}, \quad y = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$

这里 N = 2 > M = 1（实际上M=2，但rank(A)=1），但 $Ax = y$ 无解，因为第一行要求 $x_1 = 1$，第二行要求 $x_1 = 2$，矛盾。
Here N = 2 > M = 1 (actually M=2 but rank(A)=1), but $Ax = y$ has no solution because first row requires $x_1 = 1$ and second row requires $x_1 = 2$, contradiction.

**正确陈述：如果 N > M 且 rank(A) = M，则对于任何y都有解。**
**Correct statement: If N > M and rank(A) = M, then there is a solution for any y.**

---

**部分c)：如果 N > M，零空间维度 > 0 / Part c): If N > M, Nullspace Dimension > 0**

**答案：正确 / Answer: True**

**证明 / Proof:**

根据秩-零度定理：
From Rank-nullity theorem:

$$\text{rank}(A) + \text{nullity}(A) = N$$

我们也知道：
We also know:

$$\text{rank}(A) \le M$$

因此：
Thus:

$$\text{nullity}(A) = N - \text{rank}(A) \ge N - M > 0 \quad \square$$

**结论：如果 N > M，零空间维度确实大于零。**
**Conclusion: If N > M, nullspace dimension is indeed greater than zero.**

---

**部分d)：如果 N < M，某些y无解 / Part d): If N < M, Some y Have No Solution**

**答案：正确 / Answer: True**

**证明 / Proof:**

如果 N < M，则：
If N < M, then:

$$\text{rank}(A) \le \min(M, N) = N < M$$

这意味着A的列空间（$\text{col}(A)$）的维度最多为N，而y在M维空间中。
This means dimension of column space of A ($\text{col}(A)$) is at most N, while y is in M-dimensional space.

由于 $\dim(\text{col}(A)) \le N < M$，存在不在列空间中的y，使得 $Ax = y$ 无解。
Since $\dim(\text{col}(A)) \le N < M$, there exist y not in column space, making $Ax = y$ unsolvable.

---

**部分e)：如果 N < M，$Ax = 0$ 的唯一解是 $x = 0$ / Part e): If N < M, Only Solution of $Ax = 0$ is $x = 0$**

**答案：错误 / Answer: False**

**反例 / Counterexample:**
考虑：
Consider:

$$A = \begin{bmatrix} 1 & 1 \\ 1 & 1 \\ 0 & 0 \end{bmatrix}$$

这里 N = 2 < M = 3，但 $Ax = 0$ 有非零解：
Here N = 2 < M = 3, but $Ax = 0$ has nonzero solution:

$$x = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$$

**正确陈述：如果 N < M 且 rank(A) = N，则 $Ax = 0$ 的唯一解是 $x = 0$。**
**Correct statement: If N < M and rank(A) = N, then the only solution of $Ax = 0$ is $x = 0$.**

**关键词 / Keywords:** 秩-零度定理、维度、线性方程组、零空间。

### 例题3：线性回归的坐标下降法 / Coordinate Descent for Linear Regression

**题目 / Question:**  
我们希望使用坐标下降法求解以下线性回归问题：
We would like to solve the following linear regression problem using coordinate descent:

$$\min_w \frac{1}{2}\sum_{i=1}^M (y^{(i)} - w^\top x^{(i)})^2$$

其中 $w \in \mathbb{R}^{N \times 1}$, $x^{(i)} \in \mathbb{R}^{N \times 1}$。
where $w \in \mathbb{R}^{N \times 1}$ and $x^{(i)} \in \mathbb{R}^{N \times 1}$.

a) [2分] 在当前迭代中，选择 $w_k$ 进行更新。请证明以下更新规则：
a) [2 points] In the current iteration, $w_k$ is selected for update. Please prove the following update rule:

$$w_k \leftarrow \frac{\sum_{i=1}^M x_k^{(i)}(y^{(i)} - \sum_{j \neq k} w_j x_j^{(i)})}{\sum_{i=1}^M (x_k^{(i)})^2} \quad \text{(3)}$$

b) [2分] 证明以下 $w_k$ 的更新规则等价于式(3)：
b) [2 points] Prove that the following update rule for $w_k$ is equivalent to Eq. (3):

$$w \leftarrow w_k, \quad \text{(4)}$$

$$r^{(i)} \leftarrow r^{(i)} + (w - w_k) x_k^{(i)}, \quad \forall i \in \{1, 2, \ldots, M\} \quad \text{(6)}$$

其中 $r^{(i)}$ 是残差：
where $r^{(i)}$ is the residual:

$$r^{(i)} = y^{(i)} - w^\top x^{(i)} \quad \text{(7)}$$

比较两种更新规则。哪个更好？为什么？
Compare the two update rules. Which one is better and why?

**详细解答 / Detailed Solution:**

**部分a)：推导更新规则 / Part a): Derive Update Rule**

**步骤1：固定其他变量，对 $w_k$ 求导 / Step 1: Fix Other Variables, Take Derivative w.r.t. $w_k$**

目标函数：
Objective function:

$$J(w) = \frac{1}{2}\sum_{i=1}^M (y^{(i)} - w^\top x^{(i)})^2$$

固定 $w_j$ ($j \neq k$)，对 $w_k$ 求导：
Fix $w_j$ ($j \neq k$), take derivative w.r.t. $w_k$:

$$\frac{\partial J}{\partial w_k} = \sum_{i=1}^M (y^{(i)} - w^\top x^{(i)}) (-x_k^{(i)})$$

$$= -\sum_{i=1}^M x_k^{(i)} (y^{(i)} - \sum_{j=1}^N w_j x_j^{(i)})$$

**步骤2：展开并分离 $w_k$ 项 / Step 2: Expand and Separate $w_k$ Term**

$$= -\sum_{i=1}^M x_k^{(i)} \left(y^{(i)} - \sum_{j \neq k} w_j x_j^{(i)} - w_k x_k^{(i)}\right)$$

$$= -\sum_{i=1}^M x_k^{(i)} (y^{(i)} - \sum_{j \neq k} w_j x_j^{(i)}) + w_k \sum_{i=1}^M (x_k^{(i)})^2$$

**步骤3：令导数为零并求解 / Step 3: Set Derivative to Zero and Solve**

$$\frac{\partial J}{\partial w_k} = 0$$

$$-\sum_{i=1}^M x_k^{(i)} (y^{(i)} - \sum_{j \neq k} w_j x_j^{(i)}) + w_k \sum_{i=1}^M (x_k^{(i)})^2 = 0$$

$$w_k = \frac{\sum_{i=1}^M x_k^{(i)}(y^{(i)} - \sum_{j \neq k} w_j x_j^{(i)})}{\sum_{i=1}^M (x_k^{(i)})^2} \quad \square$$

---

**部分b)：证明等价性 / Part b): Prove Equivalence**

**步骤1：理解更新规则(4)和(6) / Step 1: Understand Update Rules (4) and (6)**

更新规则(4)和(6)：
Update rules (4) and (6):

$$w_k \leftarrow w_k \quad \text{(4)}$$

$$r^{(i)} \leftarrow r^{(i)} + (w_k - w_k) x_k^{(i)} = r^{(i)} + \Delta w_k \cdot x_k^{(i)} \quad \text{(6)}$$

其中 $\Delta w_k = w_k^{\text{new}} - w_k^{\text{old}}$。
where $\Delta w_k = w_k^{\text{new}} - w_k^{\text{old}}$.

**步骤2：从残差更新推导 $w_k$ 更新 / Step 2: Derive $w_k$ Update from Residual Update**

残差定义：
Residual definition:

$$r^{(i)} = y^{(i)} - w^\top x^{(i)} = y^{(i)} - \sum_{j=1}^N w_j x_j^{(i)}$$

更新后：
After update:

$$r^{(i)}_{\text{new}} = r^{(i)}_{\text{old}} + \Delta w_k x_k^{(i)}$$

$$= y^{(i)} - \sum_{j \neq k} w_j x_j^{(i)} - w_k^{\text{old}} x_k^{(i)} + \Delta w_k x_k^{(i)}$$

$$= y^{(i)} - \sum_{j \neq k} w_j x_j^{(i)} - w_k^{\text{new}} x_k^{(i)}$$

**步骤3：使用最优性条件 / Step 3: Use Optimality Condition**

在最优解处，残差与 $x_k^{(i)}$ 正交：
At optimum, residual is orthogonal to $x_k^{(i)}$:

$$\sum_{i=1}^M r^{(i)}_{\text{new}} x_k^{(i)} = 0$$

$$\sum_{i=1}^M \left(y^{(i)} - \sum_{j \neq k} w_j x_j^{(i)} - w_k^{\text{new}} x_k^{(i)}\right) x_k^{(i)} = 0$$

$$\sum_{i=1}^M x_k^{(i)} (y^{(i)} - \sum_{j \neq k} w_j x_j^{(i)}) = w_k^{\text{new}} \sum_{i=1}^M (x_k^{(i)})^2$$

$$w_k^{\text{new}} = \frac{\sum_{i=1}^M x_k^{(i)}(y^{(i)} - \sum_{j \neq k} w_j x_j^{(i)})}{\sum_{i=1}^M (x_k^{(i)})^2}$$

这与式(3)相同。
This is the same as equation (3).

**步骤4：比较两种方法 / Step 4: Compare Two Methods**

**方法1（式(3)）：直接计算 / Method 1 (Eq. 3): Direct Calculation**
- 复杂度：$O(M \cdot N^2)$（需要计算所有 $w_j x_j^{(i)}$）
- Complexity: $O(M \cdot N^2)$ (need to compute all $w_j x_j^{(i)}$)

**方法2（式(4)和(6)）：增量更新 / Method 2 (Eqs. 4 and 6): Incremental Update**
- 复杂度：$O(M \cdot N)$（只需更新残差）
- Complexity: $O(M \cdot N)$ (only need to update residuals)

**结论：方法2更好，因为计算成本更低。**
**Conclusion: Method 2 is better because it has lower computational cost.**

**关键词 / Keywords:** 残差维护、增量计算、复杂度、坐标下降、线性回归。

### 例题4：L2范数软间隔SVM / L2-Norm Soft-Margin SVM

**题目 / Question:**  
考虑使用L2范数惩罚松弛变量的软间隔SVM问题，在由核映射 $\phi(\cdot)$ 诱导的特征空间中，$K(x_i, x_j) = \phi(x_i)^\top \phi(x_j)$：
Consider the soft-margin SVM problem using an l2-norm penalty on the slack variables, in a feature space induced by a kernel map $\phi(\cdot)$ with $K(x_i, x_j) = \phi(x_i)^\top \phi(x_j)$:

$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^M \xi_i^2$$

$$\text{s.t. } y_i (w^\top \phi(x_i) + b) \ge 1 - \xi_i, \quad \forall i$$

$$\xi_i \ge 0, \quad \forall i$$

其中 $\xi_i$ 是允许第i个点违反间隔的松弛变量。
where $\xi_i$ is the slack variable that allows the i-th point to violate the margin.

a) [1分] 证明 $\xi_i \ge 0$ 的非负约束是冗余的，因此可以移除。提示：证明如果 $\xi_i < 0$ 且满足间隔约束，那么 $\xi_i = 0$ 也是一个解，且成本更低。
a) [1 point] Show that the non-negative constraint on $\xi_i$ is redundant, and hence can be dropped. Hint: show that if $\xi_i < 0$ and the margin constraint is satisfied, then $\xi_i = 0$ is also a solution with lower cost.

b) [1分] 推导拉格朗日函数。
b) [1 point] Derive the Lagrangian.

c) [1分] 推导用核K表示的SVM对偶问题。
c) [1 point] Derive the SVM dual problem in terms of the kernel K.

**详细解答 / Detailed Solution:**

**部分a)：证明非负约束冗余 / Part a): Prove Non-Negativity Constraint is Redundant**

**步骤1：假设 $\xi_i < 0$ 且满足间隔约束 / Step 1: Assume $\xi_i < 0$ and Margin Constraint Satisfied**

假设对于某个i，$\xi_i < 0$ 且满足间隔约束：
Suppose for some i, $\xi_i < 0$ and margin constraint is satisfied:

$$y_i (w^\top \phi(x_i) + b) \ge 1 - \xi_i$$

**步骤2：设置 $\xi_i = 0$ / Step 2: Set $\xi_i = 0$**

如果我们设置 $\xi_i = 0$：
If we set $\xi_i = 0$:

$$y_i (w^\top \phi(x_i) + b) \ge 1 - \xi_i > 1 = 1 - 0$$

因此间隔约束仍然满足。
Therefore margin constraint still holds.

**步骤3：比较目标值 / Step 3: Compare Objective Values**

由于目标函数包含正项 $C\xi_i^2$，设置 $\xi_i = 0$ 会减小目标值：
Since objective function includes positive term $C\xi_i^2$, setting $\xi_i = 0$ reduces objective value:

$$C \cdot 0^2 < C \cdot (\xi_i)^2 \quad \text{(因为 $\xi_i < 0$)}$$

**步骤4：得出结论 / Step 4: Conclude**

因此，任何最优解必须满足 $\xi_i \ge 0$，显式约束 $\xi_i \ge 0$ 可以移除。
Therefore, any optimal solution must satisfy $\xi_i \ge 0$, and the explicit constraint $\xi_i \ge 0$ can be dropped.

---

**部分b)：推导拉格朗日函数 / Part b): Derive Lagrangian**

**拉格朗日函数（无非负约束）/ Lagrangian (without non-negativity constraint):**

$$L(w, b, \xi, \alpha) = \frac{1}{2}\|w\|^2 + C\sum_{i=1}^M \xi_i^2 - \sum_{i=1}^M \alpha_i [y_i (w^\top \phi(x_i) + b) - 1 + \xi_i]$$

其中 $\alpha_i \ge 0$ 是与间隔约束相关的拉格朗日乘数。
where $\alpha_i \ge 0$ are Lagrange multipliers associated with margin constraints.

---

**部分c)：推导对偶问题 / Part c): Derive Dual Problem**

**步骤1：对原始变量求导并令其为零 / Step 1: Take Derivatives w.r.t. Primal Variables and Set to Zero**

对w求导：
Derivative w.r.t. w:

$$\frac{\partial L}{\partial w} = w - \sum_{i=1}^M \alpha_i y_i \phi(x_i) = 0 \quad \Rightarrow \quad w = \sum_{i=1}^M \alpha_i y_i \phi(x_i) \quad \text{(22)}$$

对b求导：
Derivative w.r.t. b:

$$\frac{\partial L}{\partial b} = -\sum_{i=1}^M \alpha_i y_i = 0 \quad \Rightarrow \quad \sum_{i=1}^M \alpha_i y_i = 0 \quad \text{(23)}$$

对 $\xi_i$ 求导：
Derivative w.r.t. $\xi_i$:

$$\frac{\partial L}{\partial \xi_i} = 2C\xi_i - \alpha_i = 0 \quad \Rightarrow \quad \xi_i = \frac{\alpha_i}{2C} \quad \text{(24)}$$

**步骤2：代入拉格朗日函数 / Step 2: Substitute into Lagrangian**

将式(22)、(23)、(24)代入拉格朗日函数：
Substitute equations (22), (23), (24) into Lagrangian:

$$L = \frac{1}{2}\|\sum_{i=1}^M \alpha_i y_i \phi(x_i)\|^2 + C\sum_{i=1}^M \left(\frac{\alpha_i}{2C}\right)^2 - \sum_{i=1}^M \alpha_i \left[y_i \left(\sum_{j=1}^M \alpha_j y_j \phi(x_j)\right)^\top \phi(x_i) + b - 1 + \frac{\alpha_i}{2C}\right]$$

**步骤3：简化 / Step 3: Simplify**

经过简化：
After simplification:

$$g(\alpha) = \sum_{i=1}^M \alpha_i - \frac{1}{2}\sum_{i=1}^M \sum_{j=1}^M \alpha_i \alpha_j y_i y_j K(x_i, x_j) - \frac{1}{4C}\sum_{i=1}^M \alpha_i^2$$

**步骤4：写出对偶问题 / Step 4: Write Dual Problem**

$$\begin{aligned}
\max_{\alpha}\;& \sum_{i=1}^M \alpha_i - \frac{1}{2}\sum_{i=1}^M \sum_{j=1}^M \alpha_i \alpha_j y_i y_j K(x_i, x_j) - \frac{1}{4C}\sum_{i=1}^M \alpha_i^2 \\
\text{s.t. }& \sum_{i=1}^M \alpha_i y_i = 0 \\
& \alpha_i \ge 0, \quad i = 1, \ldots, M
\end{aligned}$$

这是L2范数软间隔SVM的对偶形式。核技巧可以通过 $K(x_i, x_j) = \phi(x_i)^\top \phi(x_j)$ 直接应用。
This is the dual formulation of the l2-norm soft-margin SVM. The kernel trick can be directly applied through $K(x_i, x_j) = \phi(x_i)^\top \phi(x_j)$.

**关键词 / Keywords:** 可行性改进、目标减小、KKT、拉格朗日对偶、核技巧、L2范数惩罚。

