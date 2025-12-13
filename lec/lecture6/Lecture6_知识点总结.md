# Lecture 6 知识点总结 / Knowledge Summary

## Table of Contents / 目录

- [1. Support Vector Machines (SVM) / 1. 支持向量机](#1-support-vector-machines-svm--1-支持向量机)
- [2. Maximum Margin / 2. 最大间隔](#2-maximum-margin--2-最大间隔)
- [3. Hard Margin vs Soft Margin / 3. 硬间隔 vs 软间隔](#3-hard-margin-vs-soft-margin--3-硬间隔-vs-软间隔)
- [4. Kernel Trick / 4. 核技巧](#4-kernel-trick--4-核技巧)

---

## 1. Support Vector Machines (SVM) / 1. 支持向量机

#### English
Support Vector Machines (SVM) are powerful classification algorithms that find the optimal separating hyperplane by maximizing the margin between classes. They are effective for both linear and nonlinear classification.

#### 中文
支持向量机（SVM）是强大的分类算法，通过最大化类别之间的间隔来找到最优分离超平面。它们对线性和非线性分类都有效。

**基本思想 / Basic Idea:**
- 找到最优分离超平面 / Find optimal separating hyperplane
- 最大化类别之间的间隔 / Maximize margin between classes
- 使用支持向量（最接近超平面的点）/ Use support vectors (points closest to hyperplane)

**超平面方程 / Hyperplane Equation:**
$$
\mathbf{w}^\top \mathbf{x} + b = 0
$$

其中 / where:
- $\mathbf{w}$：法向量（权重向量）/ Normal vector (weight vector)
- $b$：偏置项 / Bias term

**点到超平面的距离 / Distance from Point to Hyperplane:**
$$
d = \frac{|\mathbf{w}^\top \mathbf{x} + b|}{\|\mathbf{w}\|}
$$

#### 通俗解释
SVM就像"找最宽的路"：在两个类别之间找一条最宽的"路"（间隔），让两个类别尽可能分开。支持向量是"路边的点"，决定了这条路的位置。

---

## 2. Maximum Margin / 2. 最大间隔

#### English
The margin is the distance between the hyperplane and the nearest data points. SVM maximizes this margin to find the optimal decision boundary.

#### 中文
间隔是超平面与最近数据点之间的距离。SVM最大化这个间隔以找到最优决策边界。

**函数间隔 / Functional Margin:**
$$
\hat{\gamma}^{(i)} = y^{(i)}(\mathbf{w}^\top \mathbf{x}^{(i)} + b)
$$

**几何间隔 / Geometric Margin:**
$$
\gamma^{(i)} = \frac{y^{(i)}(\mathbf{w}^\top \mathbf{x}^{(i)} + b)}{\|\mathbf{w}\|} = \frac{\hat{\gamma}^{(i)}}{\|\mathbf{w}\|}
$$

**优化问题（硬间隔）/ Optimization Problem (Hard Margin):**

原始问题 / Primal Problem:
$$
\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2
$$

约束条件 / Subject to:
$$
y^{(i)}(\mathbf{w}^\top \mathbf{x}^{(i)} + b) \ge 1, \quad i = 1, \ldots, M
$$

**对偶问题 / Dual Problem:**
$$
\max_{\boldsymbol{\alpha}} \sum_{i=1}^M \alpha_i - \frac{1}{2}\sum_{i,j=1}^M \alpha_i \alpha_j y^{(i)} y^{(j)} (\mathbf{x}^{(i)})^\top \mathbf{x}^{(j)}
$$

约束条件 / Subject to:
$$
\sum_{i=1}^M \alpha_i y^{(i)} = 0, \quad \alpha_i \ge 0
$$

**KKT条件 / KKT Conditions:**
- 互补松弛性 / Complementary slackness: $\alpha_i[y^{(i)}(\mathbf{w}^\top \mathbf{x}^{(i)} + b) - 1] = 0$
- 支持向量：$\alpha_i > 0$的点 / Support vectors: points with $\alpha_i > 0$

#### 通俗解释
最大间隔就像"找最宽的安全带"：在两个类别之间画一条线，让这条线到两边最近点的距离都最大。这样即使数据有点变化，分类也不会出错。

---

## 3. Hard Margin vs Soft Margin / 3. 硬间隔 vs 软间隔

#### English
- **Hard Margin**: Assumes data is linearly separable, no misclassifications allowed
- **Soft Margin**: Allows some misclassifications using slack variables for non-separable data

#### 中文
- **硬间隔**：假设数据线性可分，不允许任何误分类
- **软间隔**：使用松弛变量允许一些误分类，适用于不可分数据

**软间隔SVM / Soft Margin SVM:**

原始问题 / Primal Problem:
$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^M \xi_i
$$

约束条件 / Subject to:
$$
y^{(i)}(\mathbf{w}^\top \mathbf{x}^{(i)} + b) \ge 1 - \xi_i, \quad \xi_i \ge 0
$$

其中 / where:
- $\xi_i$：松弛变量 / Slack variable
- $C$：惩罚参数，控制间隔和误分类的权衡 / Penalty parameter, controls trade-off between margin and misclassification

**对偶问题 / Dual Problem:**
$$
\max_{\boldsymbol{\alpha}} \sum_{i=1}^M \alpha_i - \frac{1}{2}\sum_{i,j=1}^M \alpha_i \alpha_j y^{(i)} y^{(j)} (\mathbf{x}^{(i)})^\top \mathbf{x}^{(j)}
$$

约束条件 / Subject to:
$$
\sum_{i=1}^M \alpha_i y^{(i)} = 0, \quad 0 \le \alpha_i \le C
$$

**参数C的作用 / Role of Parameter C:**
- $C$大：更少误分类，但间隔可能较小 / Large $C$: fewer misclassifications, but smaller margin
- $C$小：允许更多误分类，但间隔较大 / Small $C$: allow more misclassifications, but larger margin

#### 通俗解释
硬间隔像"完美主义者"：要求所有点都分对，不允许任何错误。软间隔像"实用主义者"：允许一些错误，但尽量少错。参数C控制"容忍度"：C大=不能容忍错误，C小=可以容忍一些错误。

---

## 4. Kernel Trick / 4. 核技巧

#### English
The kernel trick allows SVM to handle nonlinear classification by mapping data to a higher-dimensional space without explicitly computing the transformation.

#### 中文
核技巧允许SVM通过将数据映射到高维空间来处理非线性分类，而无需显式计算变换。

**核函数 / Kernel Function:**
$$
K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) = \phi(\mathbf{x}^{(i)})^\top \phi(\mathbf{x}^{(j)})
$$

其中$\phi$是特征映射 / where $\phi$ is feature mapping

**常用核函数 / Common Kernel Functions:**

1. **线性核 / Linear Kernel:**
   $$
   K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^\top \mathbf{z}
   $$

2. **多项式核 / Polynomial Kernel:**
   $$
   K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^\top \mathbf{z} + c)^d
   $$

3. **RBF（高斯）核 / RBF (Gaussian) Kernel:**
   $$
   K(\mathbf{x}, \mathbf{z}) = \exp\left(-\gamma\|\mathbf{x} - \mathbf{z}\|^2\right)
   $$

**对偶问题（带核）/ Dual Problem (with Kernel):**
$$
\max_{\boldsymbol{\alpha}} \sum_{i=1}^M \alpha_i - \frac{1}{2}\sum_{i,j=1}^M \alpha_i \alpha_j y^{(i)} y^{(j)} K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})
$$

**预测函数 / Prediction Function:**
$$
f(\mathbf{x}) = \sum_{i=1}^M \alpha_i y^{(i)} K(\mathbf{x}^{(i)}, \mathbf{x}) + b
$$

#### 通俗解释
核技巧就像"升维魔法"：在低维空间数据不可分，但映射到高维空间就变得可分了。关键是我们可以直接计算高维空间的内积（通过核函数），而不需要真的把数据映射过去，这样计算很快。

---

## Additional Detail / 补充要点

### Support Vectors / 支持向量

#### English
Support vectors are the data points that lie on or within the margin boundaries. They determine the position of the decision boundary.

#### 中文
支持向量是位于间隔边界上或内的数据点。它们决定了决策边界的位置。

**性质 / Properties:**
- 只有支持向量的$\alpha_i > 0$ / Only support vectors have $\alpha_i > 0$
- 移除非支持向量不影响模型 / Removing non-support vectors doesn't affect model
- 支持向量数量通常很少 / Number of support vectors is usually small

---

### SMO Algorithm / SMO算法

#### English
Sequential Minimal Optimization (SMO) is an efficient algorithm for solving the SVM dual problem by optimizing two variables at a time.

#### 中文
序列最小优化（SMO）是一种高效的算法，通过每次优化两个变量来求解SVM对偶问题。

---

### Multi-class SVM / 多类SVM

#### English
- **One-vs-Rest**: Train C binary SVMs
- **One-vs-One**: Train C(C-1)/2 binary SVMs
- **Multi-class Objective**: Single optimization problem

#### 中文
- **一对多**：训练C个二元SVM
- **一对一**：训练C(C-1)/2个二元SVM
- **多类目标**：单一优化问题

---

## Learning Recommendations / 学习建议

### For Beginners / 对于初学者

1. **Understand the Geometric Intuition / 理解几何直觉:**
   - Visualize hyperplanes and margins
   - See support vectors
   - Understand maximum margin concept

2. **Implement SVM / 实现SVM:**
   - Hard margin version
   - Soft margin version
   - With different kernels

3. **Experiment with Kernels / 实验不同核:**
   - Linear kernel
   - Polynomial kernel
   - RBF kernel
   - See how they affect decision boundaries

---

## Reference Resources / 参考资源

1. **Textbooks / 教科书:**
   - Pattern Recognition and Machine Learning (Bishop)
   - The Elements of Statistical Learning (Hastie et al.)

2. **Online Resources / 在线资源:**
   - Stanford CS229 Lecture Notes
   - Scikit-learn SVM Documentation

---

## 练习题 / Practice Problems

### 问题1 / Problem 1:
推导硬间隔SVM的对偶形式，并解释支持向量的作用。
Derive the dual form of hard margin SVM and explain the role of support vectors.

### 问题2 / Problem 2:
解释软间隔SVM中参数C的作用，并说明如何选择C的值。
Explain the role of parameter C in soft margin SVM and explain how to choose the value of C.

### 问题3 / Problem 3:
比较线性核、多项式核和RBF核的特点，并说明各自的适用场景。
Compare linear, polynomial, and RBF kernels, and explain their respective use cases.

---

## 例题与解答 / Worked Examples

### 例题1：计算点到超平面的距离 / Calculating Distance from Point to Hyperplane

**题目 / Question:**  
给定超平面方程：2x₁ + 3x₂ - 6 = 0，计算点(3, 2)到该超平面的距离。
Given hyperplane equation: 2x₁ + 3x₂ - 6 = 0, calculate the distance from point (3, 2) to this hyperplane.

**详细解答 / Detailed Solution:**

**步骤1：识别超平面参数 / Step 1: Identify Hyperplane Parameters**
- 权重向量 / Weight vector: $\mathbf{w} = [2, 3]^\top$
- 偏置 / Bias: $b = -6$

**步骤2：计算未归一化距离 / Step 2: Calculate Unnormalized Distance**
$$
\mathbf{w}^\top \mathbf{x} + b = 2 \times 3 + 3 \times 2 - 6 = 6 + 6 - 6 = 6
$$

**步骤3：计算权重向量的模长 / Step 3: Calculate Weight Vector Magnitude**
$$
\|\mathbf{w}\| = \sqrt{2^2 + 3^2} = \sqrt{4 + 9} = \sqrt{13}
$$

**步骤4：计算距离 / Step 4: Calculate Distance**
$$
d = \frac{|\mathbf{w}^\top \mathbf{x} + b|}{\|\mathbf{w}\|} = \frac{|6|}{\sqrt{13}} = \frac{6}{\sqrt{13}} \approx 1.664
$$

**结论 / Conclusion:**
点(3, 2)到超平面的距离约为1.664个单位。
Distance from point (3, 2) to hyperplane is approximately 1.664 units.

---

### 例题2：软间隔SVM参数C的影响 / Effect of Parameter C in Soft Margin SVM

**题目 / Question:**  
解释软间隔SVM中参数C的作用，并说明C值大小对模型的影响。
Explain the role of parameter C in soft margin SVM and explain how C value affects the model.

**详细解答 / Detailed Solution:**

**参数C的定义 / Definition of Parameter C:**
C控制间隔大小和误分类惩罚之间的权衡 / C controls trade-off between margin size and misclassification penalty

**目标函数 / Objective Function:**
$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^M \xi_i
$$

**C值的影响 / Effect of C Value:**

1. **C很大（C → ∞）/ Large C (C → ∞):**
   - 误分类惩罚很大 / Large penalty for misclassification
   - 模型尽量不误分类，间隔可能很小 / Model tries to avoid misclassification, margin may be small
   - 接近硬间隔SVM / Approaches hard margin SVM
   - 可能过拟合 / May overfit

2. **C很小（C → 0）/ Small C (C → 0):**
   - 误分类惩罚很小 / Small penalty for misclassification
   - 允许更多误分类，间隔较大 / Allow more misclassifications, larger margin
   - 模型更简单，更平滑 / Simpler model, smoother
   - 可能欠拟合 / May underfit

3. **C适中 / Moderate C:**
   - 平衡间隔和误分类 / Balance margin and misclassification
   - 通常性能最好 / Usually best performance

**选择C的方法 / Methods to Choose C:**
- 使用交叉验证 / Use cross-validation
- 在验证集上测试不同C值 / Test different C values on validation set
- 从小的C值开始，逐步增加 / Start with small C, gradually increase

**结论 / Conclusion:**
参数C是软间隔SVM的关键超参数，需要仔细调优。通常通过交叉验证选择最优C值。
Parameter C is a key hyperparameter of soft margin SVM and needs careful tuning. Usually choose optimal C value through cross-validation.

---

