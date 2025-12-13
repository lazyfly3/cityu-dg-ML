# Assignment 4 学习笔记 / Study Notes

## 目录 / Table of Contents
1. [对称矩阵特征性质 / Spectral Properties of Symmetric Matrices](#symmetric)
2. [主成分分析 PCA / Principal Component Analysis](#pca)
3. [线性系统与卷积 / LTI Systems & Convolution](#lti)
4. [DTFT 与频域卷积 / DTFT & Frequency-Domain Convolution](#dtft)
5. [例题与解答 / Worked Examples](#worked-examples)

---

## 对称矩阵特征性质 / Spectral Properties of Symmetric Matrices <a name="symmetric"></a>

**结论 / Results:**
- 实对称矩阵的特征值均为实数。
- 不同特征值对应的特征向量正交。

**证明要点 / Proof Sketch:**
- λ ∈ ℂ, Av = λv；取共轭并用 A = Aᵀ = A*，得 ⟨Av, Av⟩ = λ ⟨v, Av⟩ = |λ|²⟨v, v⟩ ⇒ λ ∈ ℝ。
- 若 Av = λv, Aw = μw, λ ≠ μ，则 (λ−μ)⟨v, w⟩ = 0 ⇒ ⟨v,w⟩=0。

---

## 主成分分析 PCA / Principal Component Analysis <a name="pca"></a>

**目标 / Objective:** 在协方差矩阵 Σ 下寻找最大方差方向。  
$$
\max_v \; v^\top \Sigma v \quad \text{s.t. } \|v\|=1
$$

**符号说明 / Symbol Explanation:**
- $\max_v$：对向量v求最大值 / Maximize with respect to vector v
- $v$：主成分方向向量（待求）/ Principal component direction vector (to be found)
- $v^\top$：v的转置（行向量形式）/ Transpose of v (row vector form)
- $\Sigma$：数据的协方差矩阵，$\Sigma = \frac{1}{M}\sum_i (x^{(i)}-\mu)(x^{(i)}-\mu)^\top$ / Covariance matrix of data, $\Sigma = \frac{1}{M}\sum_i (x^{(i)}-\mu)(x^{(i)}-\mu)^\top$
- $v^\top \Sigma v$：在方向v上的方差（标量值）/ Variance in direction v (scalar value)
- $\|v\|=1$：约束条件，v必须是单位向量（长度为1）/ Constraint: v must be unit vector (length 1)
- $\|v\|$：向量v的模长，$\|v\| = \sqrt{v_1^2 + v_2^2 + ... + v_n^2}$ / Magnitude of vector v, $\|v\| = \sqrt{v_1^2 + v_2^2 + ... + v_n^2}$

**计算步骤 / Calculation Steps:**
1. 计算数据的协方差矩阵Σ / Calculate covariance matrix Σ of data
2. 对协方差矩阵进行特征值分解：$\Sigma = Q \Lambda Q^\top$ / Perform eigenvalue decomposition: $\Sigma = Q \Lambda Q^\top$
3. 找到最大的特征值及其对应的特征向量 / Find largest eigenvalue and corresponding eigenvector
4. 将特征向量归一化（除以其模长）得到单位向量 / Normalize eigenvector (divide by its magnitude) to get unit vector
5. 这个单位向量就是第一主成分v₁ / This unit vector is the first principal component v₁

**计算示例 / Calculation Example:**
假设有3个2维数据点：x^(1) = [1, 2], x^(2) = [2, 3], x^(3) = [3, 1]
Suppose 3 two-dimensional data points: x^(1) = [1, 2], x^(2) = [2, 3], x^(3) = [3, 1]

步骤1: 计算均值 / Calculate mean:
- μ = [(1+2+3)/3, (2+3+1)/3] = [2, 2]

步骤2: 计算协方差矩阵 / Calculate covariance matrix:
- 中心化数据 / Center data: [1-2, 2-2] = [-1, 0], [2-2, 3-2] = [0, 1], [3-2, 1-2] = [1, -1]
- Σ = (1/3) × [[-1,0],[0,1],[1,-1]]^T × [[-1,0],[0,1],[1,-1]]
- Σ = [[2/3, -1/3], [-1/3, 2/3]]

步骤3: 特征值分解 / Eigenvalue decomposition:
- 特征值: λ₁ = 1, λ₂ = 1/3
- 对应特征向量: v₁ = [1/√2, -1/√2], v₂ = [1/√2, 1/√2]

步骤4: 第一主成分（最大特征值对应的特征向量）:
- v₁ = [1/√2, -1/√2] ≈ [0.707, -0.707]

步骤5: 验证 / Verify:
- v₁^T Σ v₁ = [0.707, -0.707] × [[2/3, -1/3], [-1/3, 2/3]] × [0.707, -0.707]^T = 1 = λ₁ ✓

**结果说明 / Result Explanation:**
- 第一主成分 v₁：Σ的最大特征值对应的单位特征向量 / First principal component v₁: unit eigenvector corresponding to largest eigenvalue of Σ
- 第 K 主成分 v_K：在与 v₁,...,v_{K−1} 正交约束下，取第 K 大特征值的单位特征向量 / K-th principal component v_K: unit eigenvector corresponding to K-th largest eigenvalue, orthogonal to v₁,...,v_{K−1}

**推导要点 / Derivation Highlights:**
- 拉格朗日：L(v,λ)=vᵀΣv + λ(1−vᵀv) ⇒ Σv = λv。
- 正交约束通过拉格朗日乘子 η_j 体现，最终仍为特征向量问题。

---

## 线性系统与卷积 / LTI Systems & Convolution <a name="lti"></a>

**线性性 / Linearity:** y[x₁+x₂]=y[x₁]+y[x₂]，y[αx]=αy[x]。  
**时不变性 / Time-Invariance:** 输入平移 → 输出等量平移。

**卷积 / Convolution:** 对离散 LTI 系统，y[n] = (x * h)[n] = Σ_k x[k] h[n−k]。

**判定时不变性 / Test TI:** 检查 x[n−n₀] → y[n−n₀] 是否成立。

---

## DTFT 与频域卷积 / DTFT & Frequency-Domain Convolution <a name="dtft"></a>

**DTFT 定义 / Definition:**  
$$
X(\omega) = \sum_{n=-\infty}^{\infty} x[n] e^{-j\omega n}
$$

**符号说明 / Symbol Explanation:**
- $X(\omega)$：离散时间信号的频域表示（DTFT）/ Frequency domain representation of discrete-time signal (DTFT)
- $\omega$：数字频率，取值范围通常为$[-\pi, \pi]$ / Digital frequency, typically in range $[-\pi, \pi]$
- $\sum_{n=-\infty}^{\infty}$：对所有时间点n求和（从负无穷到正无穷）/ Sum over all time points n (from negative to positive infinity)
- $x[n]$：离散时间信号在时刻n的值 / Value of discrete-time signal at time n
- $e^{-j\omega n}$：复指数函数，$e^{-j\omega n} = \cos(\omega n) - j\sin(\omega n)$ / Complex exponential, $e^{-j\omega n} = \cos(\omega n) - j\sin(\omega n)$
- $j$：虚数单位，$j^2 = -1$ / Imaginary unit, $j^2 = -1$
- 2π周期：$X(\omega + 2\pi) = X(\omega)$ / 2π periodic: $X(\omega + 2\pi) = X(\omega)$

**计算步骤 / Calculation Steps:**
1. 对每个频率$\omega$，计算所有时间点n的贡献 / For each frequency $\omega$, calculate contribution from all time points n
2. 对每个n，计算 $x[n] \times e^{-j\omega n}$ / For each n, calculate $x[n] \times e^{-j\omega n}$
3. 将所有n的结果相加 / Sum results from all n

**计算示例 / Calculation Example:**
假设离散信号 x[n] = {1, 2, 1}（n=0,1,2），计算ω=0处的DTFT
Suppose discrete signal x[n] = {1, 2, 1} (n=0,1,2), calculate DTFT at ω=0

步骤1: 对每个n计算贡献 / For each n calculate contribution:
- n=0: x[0] × e^(-j×0×0) = 1 × e^0 = 1 × 1 = 1
- n=1: x[1] × e^(-j×0×1) = 2 × e^0 = 2 × 1 = 2
- n=2: x[2] × e^(-j×0×2) = 1 × e^0 = 1 × 1 = 1

步骤2: 求和 / Sum:
- X(0) = 1 + 2 + 1 = 4

再计算ω=π/2处的DTFT:
Calculate DTFT at ω=π/2:
- n=0: 1 × e^(-j×π/2×0) = 1 × 1 = 1
- n=1: 2 × e^(-j×π/2×1) = 2 × e^(-jπ/2) = 2 × (-j) = -2j
- n=2: 1 × e^(-j×π/2×2) = 1 × e^(-jπ) = 1 × (-1) = -1
- X(π/2) = 1 - 2j - 1 = -2j

**逆变换 / Inverse DTFT:**  
$$
x[n] = \frac{1}{2\pi}\int_{-\pi}^{\pi} X(\omega) e^{j\omega n} d\omega
$$

**符号说明 / Symbol Explanation:**
- $x[n]$：时域信号在时刻n的值 / Value of time-domain signal at time n
- $\frac{1}{2\pi}$：归一化系数 / Normalization coefficient
- $\int_{-\pi}^{\pi}$：从$-\pi$到$\pi$的定积分 / Definite integral from $-\pi$ to $\pi$
- $X(\omega)$：频域信号 / Frequency domain signal
- $e^{j\omega n}$：复指数函数（与正变换符号相反）/ Complex exponential (opposite sign from forward transform)
- $d\omega$：对频率$\omega$的积分 / Integration with respect to frequency $\omega$

**计算步骤 / Calculation Steps:**
1. 对每个时间点n，计算所有频率$\omega$的贡献 / For each time point n, calculate contribution from all frequencies $\omega$
2. 对每个$\omega$，计算 $X(\omega) \times e^{j\omega n}$ / For each $\omega$, calculate $X(\omega) \times e^{j\omega n}$
3. 对所有$\omega$从$-\pi$到$\pi$积分 / Integrate over all $\omega$ from $-\pi$ to $\pi$
4. 乘以归一化系数$\frac{1}{2\pi}$ / Multiply by normalization coefficient $\frac{1}{2\pi}$

**乘积-卷积性质 / Multiplication-Convolution Property:**
- 时域相乘 ↔ 频域卷积：DTFT{x[n]y[n]} = (1/2π)(X * Y)(ω)。
- 频域卷积定义： (X * Y)(ω) = ∫_{−π}^{π} X(θ) Y(ω−θ) dθ。

---

## 例题与解答 / Worked Examples <a name="worked-examples"></a>

### 例题1：实对称矩阵的特征值和特征向量 / Eigenvalues and Eigenvectors of Real Symmetric Matrices

**题目 / Question:**  
证明实对称矩阵 $A \in \mathbb{R}^{N \times N}$（其中 $A = A^\top$）的特征值是实值的（即不是复值的）。同时证明A的对应于不同特征值的特征向量彼此正交。
Prove that the eigenvalues of a real symmetric matrix $A \in \mathbb{R}^{N \times N}$, where $A = A^\top$, are real-valued (i.e., not complex-valued). Prove also that the eigenvectors of A corresponding to different eigenvalues are orthogonal to each other.

**详细解答 / Detailed Solution:**

**部分a)：证明特征值为实数 / Part a): Prove Eigenvalues Are Real**

**步骤1：设特征对 / Step 1: Let Eigenpair**

设 $(\lambda, v)$ 是A的任意特征对，其中 $\lambda$ 可能是复数，$v$ 是对应的特征向量。
Let $(\lambda, v)$ be any eigenpair of A, where $\lambda$ may be complex, $v$ is corresponding eigenvector.

$$Av = \lambda v$$

**步骤2：使用共轭转置 / Step 2: Use Conjugate Transpose**

由于 $A = A^\top = A^*$（实对称矩阵的共轭转置等于自身）：
Since $A = A^\top = A^*$ (conjugate transpose of real symmetric matrix equals itself):

$$\langle Av, Av \rangle = v^* A^* A v = v^* A^* \lambda v = \lambda v^* A v$$

**步骤3：计算内积 / Step 3: Calculate Inner Product**

$$\langle Av, Av \rangle = \lambda v^* A v = \lambda v^* \lambda v = \lambda^2 v^* v = |\lambda|^2 \|v\|^2$$

同时：
Also:

$$\langle Av, Av \rangle = \lambda v^* A v = \lambda \lambda^* v^* v = |\lambda|^2 \|v\|^2$$

**步骤4：比较得到结论 / Step 4: Compare to Conclude**

从 $v^* A v$ 的表达式：
From expression of $v^* A v$:

$$\lambda v^* A v = \lambda^* v^* A v$$

由于 $v^* A v$ 是标量且非零（对于非零v）：
Since $v^* A v$ is scalar and nonzero (for nonzero v):

$$\lambda = \lambda^*$$

这意味着 $\lambda$ 是实数。
This means $\lambda$ is real.

**结论：实对称矩阵的特征值都是实数。**
**Conclusion: Eigenvalues of real symmetric matrix are all real.**

---

**部分b)：证明不同特征值的特征向量正交 / Part b): Prove Eigenvectors of Different Eigenvalues Are Orthogonal**

**步骤1：设两个不同的特征对 / Step 1: Let Two Different Eigenpairs**

设 $(\lambda, x)$ 和 $(\mu, y)$ 是A的两个特征对，其中 $\lambda \neq \mu$。
Let $(\lambda, x)$ and $(\mu, y)$ be two eigenpairs of A, where $\lambda \neq \mu$.

$$Ax = \lambda x, \quad Ay = \mu y$$

**步骤2：使用内积性质 / Step 2: Use Inner Product Property**

对于任何实矩阵A和任何向量x和y：
For any real matrix A and any vectors x and y:

$$\langle Ax, y \rangle = \langle x, A^\top y \rangle$$

**步骤3：应用对称性 / Step 3: Apply Symmetry**

由于 $A = A^\top$：
Since $A = A^\top$:

$$\lambda \langle x, y \rangle = \langle \lambda x, y \rangle = \langle Ax, y \rangle = \langle x, A^\top y \rangle = \langle x, Ay \rangle = \langle x, \mu y \rangle = \mu \langle x, y \rangle$$

**步骤4：得出结论 / Step 4: Conclude**

$$(\lambda - \mu) \langle x, y \rangle = 0$$

由于 $\lambda \neq \mu$，我们必须有：
Since $\lambda \neq \mu$, we must have:

$$\langle x, y \rangle = 0$$

即特征向量x和y正交。
That is, eigenvectors x and y are orthogonal.

**结论：实对称矩阵的对应于不同特征值的特征向量彼此正交。**
**Conclusion: Eigenvectors of real symmetric matrix corresponding to different eigenvalues are orthogonal to each other.**

**关键词 / Keywords:** 共轭对称、内积正定、特征值、特征向量、正交性。

### 例题2：PCA的第K大方差方向 / K-th Largest Direction of Variance in PCA

**题目 / Question:**  
推导主成分分析（PCA）中第K大方差方向。
Derive the K-th largest direction of variance in principal component analysis (PCA).

**详细解答 / Detailed Solution:**

**步骤1：理解问题 / Step 1: Understand Problem**

在PCA中，我们寻找数据方差最大的方向。第K大方差方向 $v_K$ 对应求解优化问题：
In PCA, we seek directions of maximum variance. The K-th largest direction of variance $v_K$ corresponds to solving optimization problem:

$$\max_v v^\top \Sigma v$$

$$\text{s.t. } \|v\| = 1$$

$$v^\top v_i = 0, \quad \text{for } i = 1, 2, \ldots, K-1$$

其中 $\Sigma$ 是协方差矩阵，$v_i$ 是前 $i-1$ 个主成分。
where $\Sigma$ is covariance matrix, $v_i$ are previous $i-1$ principal components.

**步骤2：K=1的情况 / Step 2: Case K=1**

对于 $v_1$（第一主成分），优化问题简化为：
For $v_1$ (first principal component), optimization problem simplifies to:

$$\max_v v^\top \Sigma v \quad \text{s.t. } \|v\| = 1$$

**步骤3：写出拉格朗日函数 / Step 3: Write Lagrangian**

$$L(v, \lambda) = v^\top \Sigma v + \lambda(1 - \|v\|^2)$$

**步骤4：对v求导并令其为零 / Step 4: Take Derivative w.r.t. v and Set to Zero**

$$\frac{\partial L}{\partial v} = 2\Sigma v - 2\lambda v = 0 \quad \Rightarrow \quad \Sigma v = \lambda v \quad \text{(2)}$$

因此 $v_1$ 是 $\Sigma$ 的特征向量，对应特征值 $\lambda$。
Therefore $v_1$ is eigenvector of $\Sigma$ with eigenvalue $\lambda$.

**步骤5：确定特征值 / Step 5: Determine Eigenvalue**

由于约束 $\|v\| = 1$，目标函数的值恰好是：
Since constraint $\|v\| = 1$, value of objective function is precisely:

$$v^\top \Sigma v = \lambda$$

因此，最优值对应 $\lambda$ 的最大值，即 $\Sigma$ 的最大特征值。
Therefore, optimal value corresponds to maximum $\lambda$, i.e., largest eigenvalue of $\Sigma$.

**结论：$v_1$ 是 $\Sigma$ 的对应于最大特征值的单位特征向量。**
**Conclusion: $v_1$ is unit eigenvector of $\Sigma$ corresponding to largest eigenvalue.**

---

**步骤6：K>1的情况 / Step 6: Case K>1**

对于 $v_K$，拉格朗日函数为：
For $v_K$, Lagrangian is:

$$L(v_K, \lambda, \eta_1, \ldots, \eta_{K-1}) = v_K^\top \Sigma v_K + \lambda(1 - \|v_K\|^2) + \sum_{j=1}^{K-1} \eta_j v_K^\top v_j$$

可以写成：
Can be written as:

$$L = v_K^\top \Sigma v_K + \lambda(1 - \|v_K\|^2) + \sum_{j=1}^{K-1} \eta_j v_K^\top v_j$$

**步骤7：对 $v_K$ 求导 / Step 7: Take Derivative w.r.t. $v_K$**

$$\frac{\partial L}{\partial v_K} = 2\Sigma v_K - 2\lambda v_K + \sum_{j=1}^{K-1} \eta_j v_j = 0$$

**步骤8：使用正交约束 / Step 8: Use Orthogonal Constraints**

将上述方程两边同时乘以 $v_j$ ($j = 1, \ldots, K-1$)：
Multiply both sides by $v_j$ ($j = 1, \ldots, K-1$):

$$2v_j^\top \Sigma v_K - 2\lambda v_j^\top v_K + \sum_{i=1}^{K-1} \eta_i v_j^\top v_i = 0$$

注意到：
Note that:

$$v_j^\top \Sigma v_K = v_j^\top \lambda_K v_K = \lambda_K v_j^\top v_K = 0$$

（因为 $v_j$ 和 $v_K$ 正交，且 $\Sigma v_K = \lambda_K v_K$）
(because $v_j$ and $v_K$ are orthogonal, and $\Sigma v_K = \lambda_K v_K$)

从正交约束可以得出结论：
From orthogonal constraints we can conclude:

$$\eta_j = 0, \quad j = 1, \ldots, K-1$$

**步骤9：得到特征值方程 / Step 9: Get Eigenvalue Equation**

将 $\eta_j = 0$ 代回最优性方程：
Substitute $\eta_j = 0$ back into optimality equation:

$$\Sigma v_K = \lambda v_K$$

即 $v_K$ 是 $\Sigma$ 的特征向量，对应特征值 $\lambda$。
That is, $v_K$ is eigenvector of $\Sigma$ with eigenvalue $\lambda$.

**步骤10：确定第K大特征值 / Step 10: Determine K-th Largest Eigenvalue**

目标函数的值是 $\lambda$。为了最大化，我们希望最大的 $\lambda$，但必须满足约束 $v_K$ 与 $v_1, \ldots, v_{K-1}$ 正交。
Value of objective function is $\lambda$. To maximize, we want largest $\lambda$, but must respect constraint that $v_K$ is orthogonal to $v_1, \ldots, v_{K-1}$.

因此，$v_K$ 应该是 $\Sigma$ 的对应于第K大特征值的单位特征向量。
Therefore, $v_K$ should be unit eigenvector of $\Sigma$ corresponding to K-th largest eigenvalue.

**结论：PCA的第K主成分是协方差矩阵的第K大特征值对应的单位特征向量。**
**Conclusion: The K-th principal component of PCA is the unit eigenvector corresponding to the K-th largest eigenvalue of the covariance matrix.**

**关键词 / Keywords:** 拉格朗日乘子、特征分解、主成分分析、方差最大化、正交约束。

### 例题3：线性时不变系统 / Linear Time-Invariant Systems

**题目 / Question:**  
1) 假设一个离散时间线性系统对于给定输入 $x[n]$ 的输出为 $y[n]$，如图1所示。确定当输入如图2所示时的响应 $y_4[n]$。
1) Suppose that a discrete-time linear system has outputs $y[n]$ for the given inputs $x[n]$, as shown in Fig. 1. Determine the response $y_4[n]$ when the input is as shown in Fig. 2.

a) [1分] 将 $x_4[n]$ 表示为 $x_1[n]$, $x_2[n]$, $x_3[n]$ 的线性组合。
a) [1 point] Express $x_4[n]$ as a linear combination of $x_1[n]$, $x_2[n]$, and $x_3[n]$.

b) [1分] 使用系统是线性的这一事实，确定 $y_4[n]$，即对 $x_4[n]$ 的响应。
b) [1 point] Using the fact that the system is linear, determine $y_4[n]$, the response to $x_4[n]$.

c) [1分] 从图1中的输入-输出对，确定系统是否是时不变的。
c) [1 point] From the input-output pairs in Fig. 1, determine whether the system is time-invariant.

2) 确定以下两种情况中 $x[n]$ 和 $h[n]$ 的离散时间卷积。
2) Determine the discrete-time convolution of $x[n]$ and $h[n]$ for the following two cases.

a) [1分] 如图3所示。
a) [1 point] As shown in Fig. 3.

b) [1分] 如图4所示。
b) [1 point] As shown in Fig. 4.

**详细解答 / Detailed Solution:**

**部分1a)：线性组合 / Part 1a): Linear Combination**

**答案：**
$$x_4[n] = 2x_1[n] - 2x_2[n] + x_3[n]$$

**解释：** 通过观察图1和图2中的信号，可以确定 $x_4[n]$ 是 $x_1[n]$, $x_2[n]$, $x_3[n]$ 的线性组合。
**Explanation:** By observing signals in Fig. 1 and Fig. 2, $x_4[n]$ can be determined as a linear combination of $x_1[n]$, $x_2[n]$, $x_3[n]$.

---

**部分1b)：使用线性性 / Part 1b): Use Linearity**

**答案：**
$$y_4[n] = 2y_1[n] - 2y_2[n] + y_3[n]$$

**解释：** 由于系统是线性的，输入的线性组合产生输出的相同线性组合。
**Explanation:** Since system is linear, linear combination of inputs produces same linear combination of outputs.

---

**部分1c)：时不变性检查 / Part 1c): Time-Invariance Check**

**答案：系统不是时不变的。**
**Answer: The system is not time-invariant.**

**解释：** 系统不是时不变的，因为输入 $x_i[n] + x_i[n-1]$ 不产生输出 $y_i[n] + y_i[n-1]$。输入 $x_1[n] + x_1[n-1]$ 是 $x_1[n] + x_1[n-1] = x_2[n]$，我们知道它产生 $y_2[n]$。由于 $y_2[n] \neq y_1[n] + y_1[n-1]$，这个系统不是时不变的。
**Explanation:** The system is not time-invariant because an input $x_i[n] + x_i[n-1]$ does not produce an output $y_i[n] + y_i[n-1]$. The input $x_1[n] + x_1[n-1]$ is $x_1[n] + x_1[n-1] = x_2[n]$, which we are told produces $y_2[n]$. Since $y_2[n] \neq y_1[n] + y_1[n-1]$, this system is not time-invariant.

---

**部分2a)：卷积计算（图3） / Part 2a): Convolution Calculation (Fig. 3)**

**答案：**
$$y[0] = 2, \quad y[1] = 4, \quad y[2] = 6, \quad y[3] = 8, \quad y[4] = 6, \quad y[5] = 4, \quad y[6] = 2, \quad y[7] = 0$$

**计算过程：** 使用卷积公式 $y[n] = \sum_{k} x[k] h[n-k]$，对每个n值计算。
**Calculation process:** Use convolution formula $y[n] = \sum_{k} x[k] h[n-k]$, calculate for each n value.

---

**部分2b)：卷积计算（图4） / Part 2b): Convolution Calculation (Fig. 4)**

**答案：**
$$y[0] = 0, \quad y[1] = 0, \quad y[3] = 1, \quad y[5] = 1, \quad y[7] = 0$$

**计算过程：** 同样使用卷积公式，注意信号的特定形状和位置。
**Calculation process:** Similarly use convolution formula, note specific shapes and positions of signals.

**关键词 / Keywords:** 线性叠加、时不变性、离散时间卷积、LTI系统。

### 例题4：DTFT的乘积性质 / Product Property of DTFT

**题目 / Question:**  
对于离散时间信号 $x[n]$，离散时间傅里叶变换（DTFT）定义为：
For a discrete-time signal $x[n]$, the Discrete-Time Fourier Transform (DTFT) is defined as:

$$X(\omega) = \sum_{n=-\infty}^{\infty} x[n] e^{-j\omega n}$$

这是2π周期的。其逆变换为：
which is 2π-periodic. Its inverse transform is given by:

$$x[n] = \frac{1}{2\pi} \int_{-\pi}^{\pi} X(\omega) e^{j\omega n} d\omega$$

设 $x[n]$ 和 $y[n]$ 是两个离散时间信号，假设它们绝对可和。证明以下性质：
Let $x[n]$ and $y[n]$ be two discrete-time signals, and assume they are absolutely summable. Prove the following property:

$$\text{DTFT}\{x[n]y[n]\} = \frac{1}{2\pi}(X * Y)(\omega)$$

其中频域中的卷积定义为（具有2π周期性）：
where the convolution in the frequency domain is defined (with 2π-periodicity) as:

$$(X * Y)(\omega) = \int_{-\pi}^{\pi} X(\theta) Y(\omega - \theta) d\theta$$

**详细解答 / Detailed Solution:**

**步骤1：写出DTFT的定义 / Step 1: Write Definition of DTFT**

设 $z[n] = x[n]y[n]$。根据DTFT的定义：
Let $z[n] = x[n]y[n]$. By definition of DTFT:

$$Z(\omega) = \sum_{n=-\infty}^{\infty} z[n] e^{-j\omega n} = \sum_{n=-\infty}^{\infty} x[n]y[n] e^{-j\omega n}$$

**步骤2：代入y[n]的逆DTFT / Step 2: Substitute Inverse DTFT of y[n]**

使用逆DTFT公式：
Use inverse DTFT formula:

$$y[n] = \frac{1}{2\pi} \int_{-\pi}^{\pi} Y(\theta) e^{j\theta n} d\theta$$

代入：
Substitute:

$$Z(\omega) = \sum_{n=-\infty}^{\infty} x[n] \left[\frac{1}{2\pi} \int_{-\pi}^{\pi} Y(\theta) e^{j\theta n} d\theta\right] e^{-j\omega n}$$

**步骤3：交换求和与积分 / Step 3: Interchange Summation and Integration**

由于绝对可和性，可以交换求和与积分：
Due to absolute summability, can interchange summation and integration:

$$Z(\omega) = \frac{1}{2\pi} \int_{-\pi}^{\pi} Y(\theta) \left[\sum_{n=-\infty}^{\infty} x[n] e^{-j(\omega - \theta)n}\right] d\theta$$

**步骤4：识别DTFT / Step 4: Recognize DTFT**

方括号中的项是 $X(\omega - \theta)$：
Term in brackets is $X(\omega - \theta)$:

$$Z(\omega) = \frac{1}{2\pi} \int_{-\pi}^{\pi} Y(\theta) X(\omega - \theta) d\theta$$

**步骤5：写成卷积形式 / Step 5: Write in Convolution Form**

这正是频域卷积的定义：
This is exactly the definition of frequency-domain convolution:

$$Z(\omega) = \frac{1}{2\pi}(X * Y)(\omega) = \frac{1}{2\pi} \int_{-\pi}^{\pi} X(\theta) Y(\omega - \theta) d\theta$$

（注意：由于2π周期性，$X(\omega - \theta) = X(\theta)$ 在卷积中，但通常写成上面的形式）
(Note: Due to 2π-periodicity, $X(\omega - \theta) = X(\theta)$ in convolution, but usually written in form above)

**结论：**
$$\text{DTFT}\{x[n]y[n]\} = \frac{1}{2\pi}(X * Y)(\omega) \quad \square$$

**重要理解：** 时域乘积对应频域卷积（带归一化因子1/2π）。这是傅里叶变换的一个重要性质。
**Important understanding:** Time-domain product corresponds to frequency-domain convolution (with normalization factor 1/2π). This is an important property of Fourier transform.

**关键词 / Keywords:** 交换求和与积分、卷积定义、2π周期、DTFT、频域卷积。

