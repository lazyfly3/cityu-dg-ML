# Lecture 5 知识点总结 / Knowledge Summary

## Table of Contents / 目录

- [1. Linear Discriminant Analysis (LDA) / 1. 线性判别分析](#1-linear-discriminant-analysis-lda--1-线性判别分析)
- [2. Quadratic Discriminant Analysis (QDA) / 2. 二次判别分析](#2-quadratic-discriminant-analysis-qda--2-二次判别分析)
- [3. Gaussian Discriminant Analysis / 3. 高斯判别分析](#3-gaussian-discriminant-analysis--3-高斯判别分析)
- [4. Comparison with Logistic Regression / 4. 与逻辑回归的比较](#4-comparison-with-logistic-regression--4-与逻辑回归的比较)

---

## 1. Linear Discriminant Analysis (LDA) / 1. 线性判别分析

#### English
Linear Discriminant Analysis (LDA) is a generative classification method that assumes each class follows a Gaussian distribution with shared covariance matrix. It finds a linear decision boundary.

#### 中文
线性判别分析（LDA）是一种生成式分类方法，假设每个类别服从共享协方差矩阵的高斯分布。它找到线性决策边界。

**数学定义 / Mathematical Definition:**

**假设 / Assumptions:**
- 每个类别服从多元高斯分布 / Each class follows multivariate Gaussian distribution
- 所有类别共享相同的协方差矩阵 / All classes share the same covariance matrix

**类别分布 / Class Distribution:**
$$
P(\mathbf{x}|y=k) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma})
$$

其中 / where:
- $\boldsymbol{\mu}_k$：类别k的均值向量 / Mean vector for class k
- $\boldsymbol{\Sigma}$：共享的协方差矩阵 / Shared covariance matrix

**后验概率 / Posterior Probability:**
$$
P(y=k|\mathbf{x}) = \frac{P(\mathbf{x}|y=k) P(y=k)}{\sum_{j=1}^C P(\mathbf{x}|y=j) P(y=j)}
$$

**参数估计 / Parameter Estimation:**

**先验概率 / Prior Probability:**
$$
P(y=k) = \frac{M_k}{M}
$$

**均值向量 / Mean Vector:**
$$
\boldsymbol{\mu}_k = \frac{1}{M_k} \sum_{i:y^{(i)}=k} \mathbf{x}^{(i)}
$$

**共享协方差矩阵 / Shared Covariance Matrix:**
$$
\boldsymbol{\Sigma} = \frac{1}{M} \sum_{k=1}^C \sum_{i:y^{(i)}=k} (\mathbf{x}^{(i)} - \boldsymbol{\mu}_k)(\mathbf{x}^{(i)} - \boldsymbol{\mu}_k)^\top
$$

**决策边界 / Decision Boundary:**

对于二分类 / For binary classification:
$$
\log\frac{P(y=1|\mathbf{x})}{P(y=0|\mathbf{x})} = \mathbf{w}^\top \mathbf{x} + b
$$

其中 / where:
$$
\mathbf{w} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)
$$

$$
b = \log\frac{P(y=1)}{P(y=0)} - \frac{1}{2}(\boldsymbol{\mu}_1^\top \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0^\top \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_0)
$$

#### 通俗解释
LDA假设每个类别的数据都像一个"椭圆球"（高斯分布），而且所有类别的"椭圆球"形状相同（共享协方差）。它找到一条直线来分隔这些类别，就像用一把刀切蛋糕。

---

## 2. Quadratic Discriminant Analysis (QDA) / 2. 二次判别分析

#### English
Quadratic Discriminant Analysis (QDA) is similar to LDA but allows each class to have its own covariance matrix, resulting in a quadratic decision boundary.

#### 中文
二次判别分析（QDA）与LDA类似，但允许每个类别有自己的协方差矩阵，产生二次决策边界。

**数学定义 / Mathematical Definition:**

**假设 / Assumptions:**
- 每个类别服从多元高斯分布 / Each class follows multivariate Gaussian distribution
- 每个类别有自己的协方差矩阵 / Each class has its own covariance matrix

**类别分布 / Class Distribution:**
$$
P(\mathbf{x}|y=k) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

**协方差矩阵估计 / Covariance Matrix Estimation:**
$$
\boldsymbol{\Sigma}_k = \frac{1}{M_k} \sum_{i:y^{(i)}=k} (\mathbf{x}^{(i)} - \boldsymbol{\mu}_k)(\mathbf{x}^{(i)} - \boldsymbol{\mu}_k)^\top
$$

**决策边界 / Decision Boundary:**

对于二分类 / For binary classification:
$$
\log\frac{P(y=1|\mathbf{x})}{P(y=0|\mathbf{x})} = \mathbf{x}^\top \mathbf{A} \mathbf{x} + \mathbf{w}^\top \mathbf{x} + b
$$

这是一个二次函数，产生曲线决策边界 / This is a quadratic function, producing curved decision boundary

#### 通俗解释
QDA允许每个类别的"椭圆球"形状不同。决策边界不再是直线，而是曲线（椭圆、双曲线等），可以更灵活地分隔数据。

---

## 3. Gaussian Discriminant Analysis / 3. 高斯判别分析

#### English
Gaussian Discriminant Analysis (GDA) is a general term that includes both LDA and QDA. It models classes using Gaussian distributions.

#### 中文
高斯判别分析（GDA）是包括LDA和QDA的通用术语。它使用高斯分布建模类别。

**LDA vs QDA / LDA vs QDA:**

| 特性 / Feature | LDA | QDA |
|--------------|-----|-----|
| 协方差矩阵 / Covariance | 共享 / Shared | 独立 / Separate |
| 参数数量 / Parameters | 较少 / Fewer | 较多 / More |
| 决策边界 / Decision Boundary | 线性 / Linear | 二次 / Quadratic |
| 适用场景 / Use Case | 数据少时 / Small data | 数据多时 / Large data |

**选择准则 / Selection Criteria:**
- **LDA**: 当数据较少或类别协方差相似时 / When data is small or class covariances are similar
- **QDA**: 当数据较多且类别协方差明显不同时 / When data is large and class covariances differ significantly

#### 通俗解释
GDA是"用高斯分布建模类别"的总称。LDA是"简化版"（共享协方差），QDA是"完整版"（独立协方差）。选择哪个取决于数据量和类别差异。

---

## 4. Comparison with Logistic Regression / 4. 与逻辑回归的比较

#### English
Both LDA and logistic regression produce linear decision boundaries, but they make different assumptions and have different strengths.

#### 中文
LDA和逻辑回归都产生线性决策边界，但它们做出不同的假设并具有不同的优势。

**比较 / Comparison:**

| 特性 / Feature | LDA | 逻辑回归 / Logistic Regression |
|--------------|-----|---------------------------|
| 模型类型 / Model Type | 生成式 / Generative | 判别式 / Discriminative |
| 假设 / Assumptions | 高斯分布 / Gaussian | 无分布假设 / No distribution |
| 参数估计 / Parameter Estimation | 闭式解 / Closed form | 迭代优化 / Iterative |
| 数据要求 / Data Requirements | 需要更多数据 / Needs more data | 更灵活 / More flexible |
| 性能 / Performance | 数据满足假设时更好 / Better when assumptions hold | 通常更稳健 / Usually more robust |

**何时使用LDA / When to Use LDA:**
- 数据确实服从高斯分布 / Data truly follows Gaussian
- 有足够的数据估计协方差 / Have enough data to estimate covariance
- 需要快速训练 / Need fast training

**何时使用逻辑回归 / When to Use Logistic Regression:**
- 分布假设不成立 / Distribution assumptions don't hold
- 数据较少 / Small dataset
- 需要更灵活的模型 / Need more flexible model

#### 通俗解释
LDA和逻辑回归就像"不同的解题方法"：LDA假设数据是高斯分布（像椭圆球），逻辑回归不做假设（更通用）。如果假设成立，LDA可能更好；如果不确定，逻辑回归更安全。

---

## Additional Detail / 补充要点

### Dimensionality Reduction with LDA / 使用LDA降维

#### English
LDA can also be used for dimensionality reduction by finding directions that maximize class separation:
$$
\mathbf{w}^* = \arg\max_{\mathbf{w}} \frac{\mathbf{w}^\top \mathbf{S}_B \mathbf{w}}{\mathbf{w}^\top \mathbf{S}_W \mathbf{w}}
$$

where $\mathbf{S}_B$ is between-class scatter and $\mathbf{S}_W$ is within-class scatter.

#### 中文
LDA也可以用于降维，通过找到最大化类别分离的方向：
$$
\mathbf{w}^* = \arg\max_{\mathbf{w}} \frac{\mathbf{w}^\top \mathbf{S}_B \mathbf{w}}{\mathbf{w}^\top \mathbf{S}_W \mathbf{w}}
$$

其中$\mathbf{S}_B$是类间散度，$\mathbf{S}_W$是类内散度。

---

### Regularized LDA / 正则化LDA

#### English
When data is high-dimensional, regularize the covariance matrix:
$$
\boldsymbol{\Sigma}_{\text{reg}} = \alpha \boldsymbol{\Sigma} + (1-\alpha) \mathbf{I}
$$

#### 中文
当数据是高维时，正则化协方差矩阵：
$$
\boldsymbol{\Sigma}_{\text{reg}} = \alpha \boldsymbol{\Sigma} + (1-\alpha) \mathbf{I}
$$

---

### Computational Complexity / 计算复杂度

#### English
- **LDA**: $O(d^2 M + d^3)$ for training, $O(d^2)$ for prediction
- **QDA**: $O(C d^2 M + C d^3)$ for training, $O(C d^2)$ for prediction
- **Logistic Regression**: $O(d M)$ per iteration

#### 中文
- **LDA**: 训练$O(d^2 M + d^3)$，预测$O(d^2)$
- **QDA**: 训练$O(C d^2 M + C d^3)$，预测$O(C d^2)$
- **逻辑回归**: 每次迭代$O(d M)$

---

## Learning Recommendations / 学习建议

### For Beginners / 对于初学者

1. **Understand Gaussian Distributions / 理解高斯分布:**
   - Visualize 1D and 2D Gaussians
   - Understand mean and covariance
   - See how they affect shape

2. **Compare LDA and QDA / 比较LDA和QDA:**
   - Implement both
   - See decision boundaries
   - Understand when each works better

3. **Compare with Logistic Regression / 与逻辑回归比较:**
   - Run on same datasets
   - Compare performance
   - Understand tradeoffs

---

## Reference Resources / 参考资源

1. **Textbooks / 教科书:**
   - Pattern Recognition and Machine Learning (Bishop)
   - The Elements of Statistical Learning (Hastie et al.)

2. **Online Resources / 在线资源:**
   - Stanford CS229 Lecture Notes
   - Scikit-learn LDA/QDA Documentation

---

## 练习题 / Practice Problems

### 问题1 / Problem 1:
推导LDA的决策边界方程，并解释其几何意义。
Derive the decision boundary equation for LDA and explain its geometric meaning.

### 问题2 / Problem 2:
比较LDA和QDA的假设条件，并说明何时使用哪个。
Compare the assumptions of LDA and QDA, and explain when to use which.

### 问题3 / Problem 3:
证明LDA的后验概率可以写成sigmoid形式。
Prove that LDA posterior probability can be written in sigmoid form.

---

## 例题与解答 / Worked Examples

### 例题1：LDA参数估计 / LDA Parameter Estimation

**题目 / Question:**  
给定两类数据，估计LDA的参数（均值向量和共享协方差矩阵）。
Given two-class data, estimate LDA parameters (mean vectors and shared covariance matrix).

类别1数据 / Class 1 data: [1, 2], [2, 3], [3, 2]
类别2数据 / Class 2 data: [5, 6], [6, 7], [7, 6]

**详细解答 / Detailed Solution:**

**步骤1：估计均值向量 / Step 1: Estimate Mean Vectors**

类别1均值 / Class 1 mean:
$$
\boldsymbol{\mu}_1 = \frac{1}{3}[(1,2) + (2,3) + (3,2)] = \frac{1}{3}(6, 7) = (2, 2.33)
$$

类别2均值 / Class 2 mean:
$$
\boldsymbol{\mu}_2 = \frac{1}{3}[(5,6) + (6,7) + (7,6)] = \frac{1}{3}(18, 19) = (6, 6.33)
$$

**步骤2：估计共享协方差矩阵 / Step 2: Estimate Shared Covariance Matrix**

首先计算每个类别的协方差 / First calculate covariance for each class:

类别1中心化数据 / Class 1 centered data:
- (-1, -0.33), (0, 0.67), (1, -0.33)

类别1协方差 / Class 1 covariance:
$$
\boldsymbol{\Sigma}_1 = \frac{1}{3}\begin{bmatrix} 2 & 0 \\ 0 & 0.67 \end{bmatrix}
$$

类别2中心化数据 / Class 2 centered data:
- (-1, -0.33), (0, 0.67), (1, -0.33)

类别2协方差 / Class 2 covariance:
$$
\boldsymbol{\Sigma}_2 = \frac{1}{3}\begin{bmatrix} 2 & 0 \\ 0 & 0.67 \end{bmatrix}
$$

共享协方差 / Shared covariance:
$$
\boldsymbol{\Sigma} = \frac{1}{2}(\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2) = \frac{1}{3}\begin{bmatrix} 2 & 0 \\ 0 & 0.67 \end{bmatrix}
$$

**步骤3：计算决策边界 / Step 3: Calculate Decision Boundary**

权重向量 / Weight vector:
$$
\mathbf{w} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2) = \begin{bmatrix} 1.5 & 0 \\ 0 & 4.5 \end{bmatrix} \begin{bmatrix} -4 \\ -4 \end{bmatrix} = \begin{bmatrix} -6 \\ -18 \end{bmatrix}
$$

决策边界方程 / Decision boundary equation:
$$
-6x_1 - 18x_2 + b = 0
$$

---

### 例题2：LDA vs QDA选择 / Choosing LDA vs QDA

**题目 / Question:**  
解释在什么情况下应该使用LDA而不是QDA，并说明原因。
Explain when to use LDA instead of QDA and explain the reason.

**详细解答 / Detailed Solution:**

**使用LDA的情况 / When to Use LDA:**

1. **数据量较少 / Small Dataset:**
   - QDA需要估计更多参数（每个类别一个协方差矩阵）/ QDA needs more parameters (one covariance matrix per class)
   - 数据少时，估计多个协方差矩阵可能不准确 / With little data, estimating multiple covariance matrices may be inaccurate
   - LDA共享协方差矩阵，参数更少，估计更稳定 / LDA shares covariance matrix, fewer parameters, more stable estimation

2. **类别协方差相似 / Similar Class Covariances:**
   - 如果各类别的数据分布形状相似，共享协方差是合理的 / If data distributions have similar shapes, shared covariance is reasonable
   - LDA的假设更符合实际情况 / LDA assumptions match reality better

3. **计算效率 / Computational Efficiency:**
   - LDA计算更快，因为只需要估计一个协方差矩阵 / LDA is faster, only need to estimate one covariance matrix
   - 预测时计算更简单 / Simpler computation for prediction

**使用QDA的情况 / When to Use QDA:**

1. **数据量充足 / Sufficient Data:**
   - 有足够数据准确估计每个类别的协方差矩阵 / Have enough data to accurately estimate each class covariance matrix

2. **类别协方差明显不同 / Clearly Different Class Covariances:**
   - 各类别的数据分布形状差异很大 / Data distributions have very different shapes
   - 需要更灵活的模型 / Need more flexible model

**结论 / Conclusion:**
选择LDA还是QDA取决于数据量和类别协方差的相似性。通常，如果数据量少或协方差相似，选择LDA；如果数据量充足且协方差不同，选择QDA。
Choice between LDA and QDA depends on data size and similarity of class covariances. Usually, choose LDA if data is small or covariances are similar; choose QDA if data is sufficient and covariances differ.

---

