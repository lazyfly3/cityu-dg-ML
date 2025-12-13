# 机器学习考试知识点总结 / Machine Learning Exam Cheat Sheet
**A4纸正反面 / A4 Front & Back**

---

## 正面 / FRONT SIDE

---

### 1. 概率论基础 / PROBABILITY FUNDAMENTALS

**期望值 / Expected Value:**
$$E[X] = \sum_x x \cdot P(X=x)$$

**指示函数 / Indicator:**
$$I[X=a] = \begin{cases} 1 & X=a \\ 0 & \text{otherwise} \end{cases}, \quad E[I[X=a]] = P(X=a)$$

**熵 / Entropy:**
$$H(X) = -\sum_x P(x)\log_2 P(x) = -E[\log_2 P(X)]$$

**联合熵 / Joint Entropy:**
$$H(X,Y) = -\sum_{x,y} P(x,y)\log_2 P(x,y)$$

**条件熵 / Conditional Entropy:**
$$H(X|Y) = -\sum_{x,y} P(x,y)\log_2 P(X=x|Y=y)$$

**链式法则 / Chain Rule:**
$$H(X,Y) = H(Y) + H(X|Y) = H(X) + H(Y|X)$$

**互信息 / Mutual Information:**
$$I(X;Y) = H(X) - H(X|Y) = \sum_{x,y} P(x,y)\log_2\frac{P(x,y)}{P(x)P(y)}$$
- 如果X和Y独立，则I(X;Y) = 0
- If X and Y independent, then I(X;Y) = 0

**KL散度 / KL Divergence:**
$$D_{\text{KL}}(P\|Q) = \sum_x P(x)\log_2\frac{P(x)}{Q(x)} = E_P\left[\log\frac{P}{Q}\right]$$
- 非对称：$D_{\text{KL}}(P\|Q) \neq D_{\text{KL}}(Q\|P)$
- Non-symmetric: $D_{\text{KL}}(P\|Q) \neq D_{\text{KL}}(Q\|P)$

**对称KL散度（Jeffreys）/ Symmetrized KL:**
$$J(P_1,P_2) = D_{\text{KL}}(P_1\|P_2) + D_{\text{KL}}(P_2\|P_1)$$

**高斯分布对称KL（N维）/ Gaussian Symmetrized KL (N-dim):**
$$J(P_1,P_2) = \frac{1}{2}\text{tr}(\Sigma_1^{-1}\Sigma_2 + \Sigma_2^{-1}\Sigma_1 - 2I) + \frac{1}{2}(\mu_1-\mu_2)^\top(\Sigma_1^{-1}+\Sigma_2^{-1})(\mu_1-\mu_2)$$

---

### 2. 离散分布 / DISCRETE DISTRIBUTIONS

**泊松分布 / Poisson:**
$$P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad E[X] = \text{Var}(X) = \lambda$$

**泊松MLE / Poisson MLE:**
$$\lambda^* = \frac{1}{M}\sum_{i=1}^M k^{(i)} = \bar{k}$$

---

### 3. 连续分布 / CONTINUOUS DISTRIBUTIONS

**拉普拉斯分布 / Laplacian:**
$$p(x|\mu,\sigma) = \frac{1}{2\sigma}\exp\left(-\frac{|x-\mu|}{\sigma}\right)$$

**拉普拉斯MLE / Laplacian MLE:**
$$\mu^* = \text{median}\{x^{(i)}\}, \quad \sigma^* = \frac{1}{M}\sum_{i=1}^M |x^{(i)} - \mu^*|$$

**多元高斯 / Multivariate Gaussian:**
$$p(x|\mu,\Sigma) = \frac{1}{(2\pi)^{N/2}|\Sigma|^{1/2}}\exp\left(-\frac{1}{2}(x-\mu)^\top\Sigma^{-1}(x-\mu)\right)$$

---

### 4. 朴素贝叶斯 / NAIVE BAYES

**假设 / Assumption:** 特征独立 / Features independent

**联合概率 / Joint Probability:**
$$P(x,y|\theta) = P(y|\theta) \prod_{j=1}^N P(x_j|y,\theta)$$

**似然函数 / Likelihood:**
$$L(\theta) = \prod_{i=1}^M P(x^{(i)},y^{(i)}|\theta)$$

**对数似然 / Log-Likelihood:**
$$\ell(\theta) = \sum_{i=1}^M \log P(x^{(i)},y^{(i)}|\theta)$$

**预测 / Prediction:**
$$y^* = \arg\max_y P(y|x) = \arg\max_y P(y) \prod_j P(x_j|y)$$

---

### 5. 线性判别分析 / LINEAR DISCRIMINANT ANALYSIS (LDA)

**后验概率（二分类）/ Posterior (Binary):**
$$p(y=1|x) = \frac{1}{1+\exp(-\theta_0 - \theta^\top x)}$$

**其中 / Where:**
$$\theta_0 = \log\frac{\phi_1}{\phi_0} - \frac{1}{2}(\mu_1^\top\Sigma^{-1}\mu_1 - \mu_0^\top\Sigma^{-1}\mu_0)$$
$$\theta = \Sigma^{-1}(\mu_1 - \mu_0)$$

- LDA假设高斯分布，共享协方差矩阵
- LDA assumes Gaussian, shared covariance matrix

---

### 6. 优化 / OPTIMIZATION

**梯度下降 / Gradient Descent:**
$$w \leftarrow w - \alpha \nabla_w J(w)$$

**KKT条件 / KKT Conditions:**
- 平稳性 / Stationarity: $\nabla L = 0$
- 原始可行性 / Primal feasibility: $g_i(x) \le 0, h_j(x) = 0$
- 对偶可行性 / Dual feasibility: $\lambda_i \ge 0$
- 互补松弛性 / Complementary slackness: $\lambda_i g_i(x) = 0$

**拉格朗日函数 / Lagrangian:**
$$L(x,\lambda,\nu) = f(x) + \sum_i \lambda_i g_i(x) + \sum_j \nu_j h_j(x)$$

---

## 背面 / BACK SIDE

---

### 7. 支持向量机 / SUPPORT VECTOR MACHINE (SVM)

**硬间隔 / Hard Margin:**
$$\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{s.t. } y^{(i)}(w^\top x^{(i)} + b) \ge 1$$

**软间隔（L1）/ Soft Margin (L1):**
$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_i \xi_i \quad \text{s.t. } y^{(i)}(w^\top x^{(i)} + b) \ge 1-\xi_i, \xi_i \ge 0$$

**软间隔（L2）/ Soft Margin (L2):**
$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_i \xi_i^2 \quad \text{s.t. } y^{(i)}(w^\top x^{(i)} + b) \ge 1-\xi_i$$

**对偶问题（硬间隔）/ Dual (Hard Margin):**
$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y^{(i)}y^{(j)}x^{(i)\top}x^{(j)}$$
$$\text{s.t. } \sum_i \alpha_i y^{(i)} = 0, \quad \alpha_i \ge 0$$

**对偶问题（软间隔L1）/ Dual (Soft Margin L1):**
$$\text{s.t. } \sum_i \alpha_i y^{(i)} = 0, \quad 0 \le \alpha_i \le C$$

**对偶问题（软间隔L2）/ Dual (Soft Margin L2):**
$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y^{(i)}y^{(j)}K(x_i,x_j) - \frac{1}{4C}\sum_i\alpha_i^2$$
$$\text{s.t. } \sum_i \alpha_i y^{(i)} = 0, \quad \alpha_i \ge 0$$

**点到超平面距离 / Distance to Hyperplane:**
$$d = \frac{|w^\top x^{(i)} + b|}{\|w\|}$$

**支持向量 / Support Vectors:**
$$\alpha_i > 0 \Rightarrow y^{(i)}(w^\top x^{(i)} + b) = 1$$

---

### 8. 线性回归 / LINEAR REGRESSION

**目标 / Objective:**
$$\min_w \frac{1}{2}\sum_{i=1}^M (y^{(i)} - w^\top x^{(i)})^2$$

**坐标下降更新 / Coordinate Descent Update:**
$$w_k \leftarrow \frac{\sum_{i=1}^M x_k^{(i)}(y^{(i)} - \sum_{j\neq k} w_j x_j^{(i)})}{\sum_{i=1}^M (x_k^{(i)})^2}$$

**残差更新（高效）/ Residual Update (Efficient):**
$$r^{(i)} \leftarrow r^{(i)} + (w_k^{\text{new}} - w_k^{\text{old}}) x_k^{(i)}$$
- 复杂度：$O(MN)$ vs 直接计算 $O(MN^2)$
- Complexity: $O(MN)$ vs direct $O(MN^2)$

---

### 9. 矩阵恒等式 / MATRIX IDENTITIES

**Woodbury恒等式 / Woodbury Identity:**
$$(P^{-1} + B^\top R^{-1}B)^{-1}B^\top R^{-1} = PB^\top(BPB^\top + R)^{-1}$$
- 当 $m \ll n$ 时，右边更高效 / When $m \ll n$, right side more efficient

**特殊形式 / Special Case:**
$$(I + AB)^{-1}A = A(I + BA)^{-1}$$

**迹的性质 / Trace Properties:**
$$\text{tr}(AB) = \text{tr}(BA), \quad \text{tr}(ABC) = \text{tr}(CAB) = \text{tr}(BCA)$$

---

### 10. 线性系统 / LINEAR SYSTEMS

**秩-零度定理 / Rank-Nullity Theorem:**
$$\text{rank}(A) + \text{nullity}(A) = N$$

**解的存在性 / Solution Existence:**
- $N = M$ 且 $A$ 可逆：唯一解 / Unique solution if invertible
- $N > M$ 且 $\text{rank}(A) = M$：总是有解 / Always solvable
- $N > M$：$\text{nullity}(A) \ge N - M > 0$
- $N < M$：某些 $y$ 无解 / Some $y$ have no solution

---

### 11. 主成分分析 / PRINCIPAL COMPONENT ANALYSIS (PCA)

**目标 / Objective:**
$$\max_v v^\top\Sigma v \quad \text{s.t. } \|v\| = 1$$

**解 / Solution:**
$$v_1 = \text{最大特征值对应的单位特征向量 / Unit eigenvector of largest eigenvalue}$$

**第K主成分 / K-th Principal Component:**
$$\max_v v^\top\Sigma v \quad \text{s.t. } \|v\|=1, v^\top v_i=0 \text{ for } i=1,\ldots,K-1$$

**解：第K大特征值对应的特征向量**
**Solution: Eigenvector of K-th largest eigenvalue**

---

### 12. 对称矩阵 / SYMMETRIC MATRICES

**性质 / Properties:**
- 特征值都是实数 / All eigenvalues are real
- 不同特征值的特征向量正交 / Eigenvectors of different eigenvalues are orthogonal
- $A = A^\top$：对称 / Symmetric

**证明要点 / Proof Sketch:**
- $\lambda = \lambda^*$（共轭相等）$\Rightarrow \lambda \in \mathbb{R}$
- $(\lambda - \mu)\langle v_1, v_2 \rangle = 0$ 且 $\lambda \neq \mu \Rightarrow \langle v_1, v_2 \rangle = 0$

---

### 13. 信号处理 / SIGNAL PROCESSING

**离散时间卷积 / Discrete-Time Convolution:**
$$y[n] = x[n] * h[n] = \sum_k x[k]h[n-k]$$

**LTI系统 / LTI System:**
- 线性性 / Linearity: $T(ax_1 + bx_2) = aT(x_1) + bT(x_2)$
- 时不变性 / Time-invariance: $y[n-k] = T(x[n-k])$

**DTFT / Discrete-Time Fourier Transform:**
$$X(\omega) = \sum_{n=-\infty}^{\infty} x[n]e^{-j\omega n}$$

**逆DTFT / Inverse DTFT:**
$$x[n] = \frac{1}{2\pi}\int_{-\pi}^{\pi} X(\omega)e^{j\omega n}d\omega$$

**时域乘积 $\Leftrightarrow$ 频域卷积 / Time Product $\Leftrightarrow$ Frequency Convolution:**
$$\text{DTFT}\{x[n]y[n]\} = \frac{1}{2\pi}(X * Y)(\omega)$$

**频域卷积 / Frequency Convolution:**
$$(X * Y)(\omega) = \int_{-\pi}^{\pi} X(\theta)Y(\omega-\theta)d\theta$$

---

### 14. 重要不等式 / IMPORTANT INEQUALITIES

**Jensen不等式 / Jensen's Inequality:**
$$E[f(X)] \ge f(E[X]) \text{ if } f \text{ convex}$$
$$E[f(X)] \le f(E[X]) \text{ if } f \text{ concave}$$

**Cauchy-Schwarz / Cauchy-Schwarz:**
$$|\langle x,y \rangle| \le \|x\|\|y\|$$

---

### 15. 记忆要点 / KEY MEMORY POINTS

**最大似然估计 / MLE:**
- 泊松：$\lambda^* = \bar{k}$（样本均值）
- 拉普拉斯：$\mu^* = \text{median}$, $\sigma^* = \text{mean absolute deviation}$

**信息论 / Information Theory:**
- 熵：不确定性度量 / Measures uncertainty
- 互信息：相关性度量 / Measures correlation
- KL散度：分布差异度量 / Measures distribution difference

**优化 / Optimization:**
- 梯度下降：$\nabla J = 0$ 找到最优点 / Find optimum
- KKT：约束优化的必要条件 / Necessary conditions for constrained optimization

**SVM关键差异 / SVM Key Differences:**
- 硬间隔：$\alpha_i \ge 0$
- 软间隔L1：$0 \le \alpha_i \le C$
- 软间隔L2：$\alpha_i \ge 0$（无上界，但有二次项）

**PCA关键点 / PCA Key Points:**
- 最大化方差方向 / Maximize variance direction
- 协方差矩阵的特征向量 / Eigenvectors of covariance matrix
- 主成分彼此正交 / Principal components are orthogonal

---

**提示 / TIPS:**
- 检查单位：确保公式维度一致 / Check units: ensure consistent dimensions
- 边界情况：检查 $\lambda > 0$, $\sigma > 0$, $\alpha \ge 0$ 等 / Boundary cases
- 对称性：利用对称性简化计算 / Use symmetry
- 数值稳定性：使用对数避免下溢 / Use logs for numerical stability

---

*打印提示：使用小字体（9-10pt），单倍行距，双面打印*
*Printing tip: Use small font (9-10pt), single spacing, double-sided*

