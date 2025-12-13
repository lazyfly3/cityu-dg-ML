# Lecture 9 知识点总结 / Knowledge Summary

## Table of Contents / 目录

- [1. Principal Component Analysis (PCA) / 1. 主成分分析](#1-principal-component-analysis-pca--1-主成分分析)
- [2. Dimensionality Reduction / 2. 降维](#2-dimensionality-reduction--2-降维)
- [3. Eigenvalue Decomposition / 3. 特征值分解](#3-eigenvalue-decomposition--3-特征值分解)
- [4. Applications / 4. 应用](#4-applications--4-应用)

---

## 1. Principal Component Analysis (PCA) / 1. 主成分分析

#### English
Principal Component Analysis (PCA) is an unsupervised dimensionality reduction technique that finds directions of maximum variance in the data and projects data onto these directions.

#### 中文
主成分分析（PCA）是一种无监督降维技术，找到数据中方差最大的方向，并将数据投影到这些方向上。

**数学定义 / Mathematical Definition:**

**目标 / Objective:**
找到单位向量$\mathbf{v}$，使投影数据的方差最大 / Find unit vector $\mathbf{v}$ that maximizes variance of projected data

**优化问题 / Optimization Problem:**
$$
\max_{\mathbf{v}} \mathbf{v}^\top \boldsymbol{\Sigma} \mathbf{v} \quad \text{s.t. } \|\mathbf{v}\| = 1
$$

其中 / where:
- $\boldsymbol{\Sigma} = \frac{1}{M}\sum_{i=1}^M (\mathbf{x}^{(i)} - \boldsymbol{\mu})(\mathbf{x}^{(i)} - \boldsymbol{\mu})^\top$：协方差矩阵 / Covariance matrix
- $\boldsymbol{\mu} = \frac{1}{M}\sum_{i=1}^M \mathbf{x}^{(i)}$：均值向量 / Mean vector

**拉格朗日函数 / Lagrangian:**
$$
L(\mathbf{v}, \lambda) = \mathbf{v}^\top \boldsymbol{\Sigma} \mathbf{v} + \lambda(1 - \mathbf{v}^\top \mathbf{v})
$$

**最优解 / Optimal Solution:**
对$\mathbf{v}$求导并令其为零 / Take derivative w.r.t. $\mathbf{v}$ and set to zero:
$$
\boldsymbol{\Sigma} \mathbf{v} = \lambda \mathbf{v}
$$

因此，$\mathbf{v}$是$\boldsymbol{\Sigma}$的特征向量，$\lambda$是对应的特征值 / Therefore, $\mathbf{v}$ is eigenvector of $\boldsymbol{\Sigma}$, $\lambda$ is corresponding eigenvalue

**第一主成分 / First Principal Component:**
对应最大特征值的特征向量 / Eigenvector corresponding to largest eigenvalue

#### 通俗解释
PCA就像"找最重要的方向"：数据可能有很多维度，但大部分信息可能集中在少数几个方向上。PCA找到这些"最重要的方向"（主成分），把数据投影到这些方向上，减少维度但保留大部分信息。

---

## 2. Dimensionality Reduction / 2. 降维

#### English
Dimensionality reduction reduces the number of features while preserving as much information as possible. PCA achieves this by projecting data onto lower-dimensional subspaces.

#### 中文
降维在尽可能保留信息的同时减少特征数量。PCA通过将数据投影到低维子空间来实现这一点。

**投影 / Projection:**

将数据投影到前$k$个主成分 / Project data onto first $k$ principal components:
$$
\mathbf{z}^{(i)} = \mathbf{V}_k^\top (\mathbf{x}^{(i)} - \boldsymbol{\mu})
$$

其中 / where:
- $\mathbf{V}_k = [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k]$：前$k$个主成分（列向量）/ First $k$ principal components (column vectors)
- $\mathbf{z}^{(i)}$：降维后的数据 / Reduced-dimensional data

**重建 / Reconstruction:**
$$
\hat{\mathbf{x}}^{(i)} = \boldsymbol{\mu} + \mathbf{V}_k \mathbf{z}^{(i)}
$$

**方差保留 / Variance Preserved:**
保留的方差比例 / Proportion of variance preserved:
$$
\frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^d \lambda_j}
$$

其中$\lambda_j$是第$j$个特征值 / where $\lambda_j$ is $j$-th eigenvalue

**计算步骤 / Calculation Steps:**
1. 中心化数据：$\mathbf{x}^{(i)} \leftarrow \mathbf{x}^{(i)} - \boldsymbol{\mu}$ / Center data
2. 计算协方差矩阵$\boldsymbol{\Sigma}$ / Compute covariance matrix
3. 特征值分解：$\boldsymbol{\Sigma} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^\top$ / Eigenvalue decomposition
4. 选择前$k$个特征向量 / Select first $k$ eigenvectors
5. 投影数据 / Project data

#### 通俗解释
降维就像"压缩"：把高维数据压缩到低维，但尽量保留重要信息。PCA找到"信息最多"的方向，只保留这些方向，丢弃信息少的方向。

---

## 3. Eigenvalue Decomposition / 3. 特征值分解

#### English
Eigenvalue decomposition factorizes a matrix into eigenvectors and eigenvalues, which is fundamental to PCA and many other techniques.

#### 中文
特征值分解将矩阵分解为特征向量和特征值，这是PCA和许多其他技术的基础。

**特征值分解 / Eigenvalue Decomposition:**
$$
\boldsymbol{\Sigma} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^\top
$$

其中 / where:
- $\mathbf{V} = [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_d]$：特征向量矩阵（列向量）/ Eigenvector matrix (column vectors)
- $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_d)$：特征值对角矩阵 / Eigenvalue diagonal matrix
- $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_d$：特征值（降序排列）/ Eigenvalues (in descending order)

**性质 / Properties:**
- 特征向量正交：$\mathbf{v}_i^\top \mathbf{v}_j = 0$（$i \ne j$）/ Eigenvectors orthogonal
- 特征向量归一化：$\|\mathbf{v}_i\| = 1$ / Eigenvectors normalized
- 特征值非负（对于协方差矩阵）/ Eigenvalues non-negative (for covariance matrix)

**计算特征值和特征向量 / Computing Eigenvalues and Eigenvectors:**
求解特征方程 / Solve characteristic equation:
$$
\det(\boldsymbol{\Sigma} - \lambda \mathbf{I}) = 0
$$

对于每个特征值$\lambda_i$，求解 / For each eigenvalue $\lambda_i$, solve:
$$
(\boldsymbol{\Sigma} - \lambda_i \mathbf{I}) \mathbf{v}_i = \mathbf{0}
$$

#### 通俗解释
特征值分解就像"找主方向"：协方差矩阵告诉我们数据在各个方向上的"伸展程度"。特征向量是"主方向"，特征值是"伸展程度"。PCA就是找"伸展最大"的方向。

---

## 4. Applications / 4. 应用

#### English
PCA has many applications including data visualization, noise reduction, feature extraction, and data compression.

#### 中文
PCA有许多应用，包括数据可视化、降噪、特征提取和数据压缩。

**数据可视化 / Data Visualization:**
将高维数据投影到2D或3D进行可视化 / Project high-dimensional data to 2D or 3D for visualization

**降噪 / Noise Reduction:**
保留主要成分，去除噪声 / Keep principal components, remove noise

**特征提取 / Feature Extraction:**
从原始特征中提取更有意义的特征 / Extract more meaningful features from original features

**数据压缩 / Data Compression:**
减少存储空间和计算成本 / Reduce storage space and computational cost

**预处理 / Preprocessing:**
在机器学习模型之前使用PCA降维 / Use PCA for dimensionality reduction before ML models

#### 通俗解释
PCA的应用就像"多用途工具"：
- **可视化**：把高维数据画在纸上（2D/3D）
- **降噪**：保留信号，去除噪声
- **压缩**：用更少的数据表示同样的信息
- **加速**：减少计算量

---

## Additional Detail / 补充要点

### Singular Value Decomposition (SVD) / 奇异值分解

#### English
SVD is an alternative to eigenvalue decomposition that works directly on the data matrix:
$$
\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top
$$

PCA can be computed using SVD, which is more numerically stable.

#### 中文
SVD是特征值分解的替代方法，直接对数据矩阵操作：
$$
\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top
$$

PCA可以使用SVD计算，数值上更稳定。

---

### Choosing Number of Components / 选择主成分数量

#### English
- **Variance threshold**: Keep components until cumulative variance exceeds threshold (e.g., 95%)
- **Scree plot**: Look for "elbow" in plot of eigenvalues
- **Cross-validation**: Use validation set to choose optimal $k$

#### 中文
- **方差阈值**：保留主成分直到累积方差超过阈值（如95%）
- **碎石图**：在特征值图中寻找"肘部"
- **交叉验证**：使用验证集选择最优$k$

---

### Limitations / 局限性

#### English
- **Linear**: Only captures linear relationships
- **Global**: Assumes same structure everywhere
- **Sensitive to scaling**: Features should be standardized
- **Interpretability**: Principal components may not be interpretable

#### 中文
- **线性**：只捕捉线性关系
- **全局**：假设处处结构相同
- **对缩放敏感**：特征应该标准化
- **可解释性**：主成分可能不可解释

---

## Learning Recommendations / 学习建议

### For Beginners / 对于初学者

1. **Understand the Geometry / 理解几何:**
   - Visualize 2D data
   - See how PCA finds principal directions
   - Understand projection

2. **Implement PCA / 实现PCA:**
   - From scratch using eigenvalue decomposition
   - Using SVD
   - Compare with sklearn implementation

3. **Apply to Real Data / 应用到真实数据:**
   - Image data
   - Text data
   - High-dimensional datasets
   - Visualize results

---

## Reference Resources / 参考资源

1. **Textbooks / 教科书:**
   - Pattern Recognition and Machine Learning (Bishop)
   - The Elements of Statistical Learning (Hastie et al.)

2. **Online Resources / 在线资源:**
   - Stanford CS229 Lecture Notes
   - Scikit-learn PCA Documentation

---

## 练习题 / Practice Problems

### 问题1 / Problem 1:
推导PCA的第一主成分，并解释为什么是协方差矩阵的最大特征值对应的特征向量。
Derive the first principal component of PCA and explain why it is the eigenvector corresponding to the largest eigenvalue of the covariance matrix.

### 问题2 / Problem 2:
解释如何选择PCA的主成分数量，并比较不同的选择方法。
Explain how to choose the number of principal components in PCA and compare different selection methods.

### 问题3 / Problem 3:
比较PCA和线性判别分析（LDA）的异同。
Compare the similarities and differences between PCA and Linear Discriminant Analysis (LDA).

---

## 例题与解答 / Worked Examples

### 例题1：PCA计算 / PCA Calculation

**题目 / Question:**  
给定二维数据点：(1, 2), (2, 3), (3, 1), (4, 4)，计算第一主成分。
Given 2D data points: (1, 2), (2, 3), (3, 1), (4, 4), calculate the first principal component.

**详细解答 / Detailed Solution:**

**步骤1：计算均值 / Step 1: Calculate Mean**
$$
\boldsymbol{\mu} = \frac{1}{4}[(1,2) + (2,3) + (3,1) + (4,4)] = (2.5, 2.5)
$$

**步骤2：中心化数据 / Step 2: Center Data**
- (-1.5, -0.5), (-0.5, 0.5), (0.5, -1.5), (1.5, 1.5)

**步骤3：计算协方差矩阵 / Step 3: Calculate Covariance Matrix**
$$
\boldsymbol{\Sigma} = \frac{1}{4}\begin{bmatrix} 2.5 & 1.0 \\ 1.0 & 1.5 \end{bmatrix}
$$

**步骤4：特征值分解 / Step 4: Eigenvalue Decomposition**
特征值 / Eigenvalues: $\lambda_1 \approx 3.0$, $\lambda_2 \approx 0.75$
特征向量 / Eigenvectors: 
- $\mathbf{v}_1 \approx [0.866, 0.5]^\top$ (对应 $\lambda_1$)
- $\mathbf{v}_2 \approx [-0.5, 0.866]^\top$ (对应 $\lambda_2$)

**步骤5：第一主成分 / Step 5: First Principal Component**
第一主成分是最大特征值对应的归一化特征向量：
The first principal component is the normalized eigenvector corresponding to largest eigenvalue:
$$
\mathbf{v}_1 = \frac{[0.866, 0.5]^\top}{\|[0.866, 0.5]\|} \approx [0.866, 0.5]^\top
$$

**结论 / Conclusion:**
第一主成分方向约为[0.866, 0.5]，这是数据方差最大的方向。
First principal component direction is approximately [0.866, 0.5], which is the direction of maximum variance.

---

### 例题2：PCA降维 / PCA Dimensionality Reduction

**题目 / Question:**  
解释如何使用PCA将数据从d维降到k维（k < d），并说明保留的方差比例。
Explain how to use PCA to reduce data from d dimensions to k dimensions (k < d), and explain the proportion of variance preserved.

**详细解答 / Detailed Solution:**

**步骤1：特征值分解 / Step 1: Eigenvalue Decomposition**
对协方差矩阵进行特征值分解 / Perform eigenvalue decomposition on covariance matrix:
$$
\boldsymbol{\Sigma} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^\top
$$

特征值按降序排列：$\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_d$ / Eigenvalues in descending order: $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_d$

**步骤2：选择前k个主成分 / Step 2: Select First k Principal Components**
选择前k个特征向量 / Select first k eigenvectors:
$$
\mathbf{V}_k = [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k]
$$

**步骤3：投影数据 / Step 3: Project Data**
将数据投影到k维空间 / Project data to k-dimensional space:
$$
\mathbf{z}^{(i)} = \mathbf{V}_k^\top (\mathbf{x}^{(i)} - \boldsymbol{\mu})
$$

**步骤4：计算保留的方差比例 / Step 4: Calculate Proportion of Variance Preserved**
$$
\text{保留方差比例} = \frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^d \lambda_j}
$$

**示例 / Example:**
假设d=10，k=2，特征值为：
Suppose d=10, k=2, eigenvalues:
- $\lambda_1 = 5.0$, $\lambda_2 = 3.0$, $\lambda_3 = 1.0$, $\ldots$, $\lambda_{10} = 0.1$

保留的方差比例 / Proportion of variance preserved:
$$
\frac{5.0 + 3.0}{5.0 + 3.0 + 1.0 + \ldots + 0.1} = \frac{8.0}{10.0} = 0.8 = 80\%
$$

**结论 / Conclusion:**
PCA通过保留前k个主成分，可以保留大部分方差（通常80-95%），同时将维度从d降到k。
PCA can preserve most variance (usually 80-95%) by keeping first k principal components while reducing dimensions from d to k.

---

