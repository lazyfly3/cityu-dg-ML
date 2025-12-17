# Lecture 3 知识点总结 / Knowledge Summary

## Table of Contents / 目录

- [1. Logistic Regression / 1. 逻辑回归](#1-logistic-regression--1-逻辑回归)
- [2. Classification / 2. 分类问题](#2-classification--2-分类问题)
- [3. Decision Boundaries / 3. 决策边界](#3-decision-boundaries--3-决策边界)
- [4. Multiclass Classification / 4. 多类分类](#4-multiclass-classification--4-多类分类)

---

## 1. Logistic Regression / 1. 逻辑回归

#### English
Logistic regression is a classification algorithm that models the probability of a binary outcome using the logistic (sigmoid) function. It extends linear regression to classification problems.

#### 中文
逻辑回归是一种分类算法，使用逻辑（sigmoid）函数建模二元结果的概率。它将线性回归扩展到分类问题。

**数学定义 / Mathematical Definition:**

**Sigmoid函数 / Sigmoid Function:**

$$
\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}
$$

**逻辑回归模型 / Logistic Regression Model:**

$$
P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}
$$

$$
P(y=0 \mid \mathbf{x}) = 1 - P(y=1 \mid \mathbf{x}) = \frac{e^{-(\mathbf{w}^\top \mathbf{x} + b)}}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}
$$

**符号说明 / Symbol Explanation:**
- $P(y=1 \mid \mathbf{x})$：在给定特征$\mathbf{x}$下，类别为1的概率 / Probability of class 1 given features $\mathbf{x}$
- $\sigma(z)$：Sigmoid函数，将实数映射到(0,1)区间 / Sigmoid function, maps real numbers to (0,1)
- $\mathbf{w}$：权重向量 / Weight vector
- $b$：偏置项 / Bias term

**计算步骤 / Calculation Steps:**
1. 计算线性组合：$z = \mathbf{w}^\top \mathbf{x} + b$ / Calculate linear combination
2. 应用sigmoid函数：$P(y=1) = \sigma(z)$ / Apply sigmoid function
3. 预测类别：如果$P(y=1) > 0.5$，预测类别1，否则预测类别0 / Predict class: if $P(y=1) > 0.5$, predict class 1, else predict class 0

#### 通俗解释
逻辑回归就像"打分后转概率"：先像线性回归一样计算一个分数，然后用sigmoid函数把这个分数转换成0到1之间的概率。概率>0.5就预测一类，否则预测另一类。就像考试：分数高（>50分）就及格，否则不及格。

---

## 2. Classification / 2. 分类问题

#### English
Classification is a supervised learning task where the goal is to predict discrete class labels. The output is categorical rather than continuous.

#### 中文
分类是一种监督学习任务，目标是预测离散的类别标签。输出是分类的而不是连续的。

**损失函数：交叉熵 / Loss Function: Cross-Entropy**

**二元交叉熵 / Binary Cross-Entropy:**

$$
J(\mathbf{w}, b) = -\frac{1}{M}\sum_{i=1}^M \left[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})\right]
$$

其中 / where:
- $y^{(i)} \in \{0, 1\}$：真实标签 / True label
- $\hat{y}^{(i)} = P(y=1 \mid \mathbf{x}^{(i)})$：预测概率 / Predicted probability

**梯度计算 / Gradient Calculation:**

对权重 / For weights:

$$
\frac{\partial J}{\partial w_j} = \frac{1}{M}\sum_{i=1}^M (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}
$$

对偏置 / For bias:

$$
\frac{\partial J}{\partial b} = \frac{1}{M}\sum_{i=1}^M (\hat{y}^{(i)} - y^{(i)})
$$

**计算步骤 / Calculation Steps:**
1. 初始化权重和偏置 / Initialize weights and bias
2. 对每个样本计算预测概率 / For each sample, calculate predicted probability
3. 计算交叉熵损失 / Calculate cross-entropy loss
4. 计算梯度并更新参数 / Calculate gradient and update parameters

#### 通俗解释
分类就像"贴标签"：给每个东西贴上正确的类别标签。交叉熵损失衡量"预测概率"和"真实标签"的差距：预测越准确，损失越小。

---

## 3. Decision Boundaries / 3. 决策边界

#### English
A decision boundary is the surface that separates different classes in the feature space. For logistic regression, the decision boundary is linear.

#### 中文
决策边界是在特征空间中分隔不同类别的表面。对于逻辑回归，决策边界是线性的。

**决策边界方程 / Decision Boundary Equation:**

当 / When $P(y=1 \mid \mathbf{x}) = P(y=0 \mid \mathbf{x}) = 0.5$:

$$
\mathbf{w}^\top \mathbf{x} + b = 0
$$

**几何解释 / Geometric Interpretation:**
- 决策边界是一个超平面 / Decision boundary is a hyperplane
- 在二维空间中是一条直线 / In 2D space, it's a line
- 在三维空间中是一个平面 / In 3D space, it's a plane

**计算示例 / Calculation Example:**
假设 / Suppose $\mathbf{w} = [2, -1]$, $b = 3$:
- 决策边界：$2x_1 - x_2 + 3 = 0$，即 $x_2 = 2x_1 + 3$
- 在直线上方：$2x_1 - x_2 + 3 > 0$，预测类别1 / Above line: predict class 1
- 在直线下方：$2x_1 - x_2 + 3 < 0$，预测类别0 / Below line: predict class 0

#### 通俗解释
决策边界就像"分界线"：把不同类别的数据分开。逻辑回归的决策边界是直线（或平面），就像用一条线把两类点分开。

---

## 4. Multiclass Classification / 4. 多类分类

#### English
Multiclass classification extends binary classification to more than two classes. Common approaches include one-vs-rest and softmax regression.

#### 中文
多类分类将二元分类扩展到两个以上的类别。常见方法包括一对多和softmax回归。

**Softmax回归 / Softmax Regression:**

**Softmax函数 / Softmax Function:**

$$
\text{softmax}(z_j) = \frac{e^{z_j}}{\sum_{k=1}^C e^{z_k}}
$$

**多类逻辑回归模型 / Multiclass Logistic Regression Model:**

$$
P(y=j \mid \mathbf{x}) = \frac{e^{\mathbf{w}_j^\top \mathbf{x} + b_j}}{\sum_{k=1}^C e^{\mathbf{w}_k^\top \mathbf{x} + b_k}}
$$

其中 / where:
- $C$：类别数量 / Number of classes
- $\mathbf{w}_j$：第j类的权重向量 / Weight vector for class j
- $b_j$：第j类的偏置 / Bias for class j

**交叉熵损失（多类）/ Cross-Entropy Loss (Multiclass):**

$$
J(\mathbf{W}, \mathbf{b}) = -\frac{1}{M}\sum_{i=1}^M \sum_{j=1}^C y_j^{(i)}\log(\hat{y}_j^{(i)})
$$

其中 / where:
- $y_j^{(i)}$：如果样本i属于类别j则为1，否则为0 / 1 if sample i belongs to class j, else 0
- $\hat{y}_j^{(i)} = P(y=j \mid \mathbf{x}^{(i)})$：预测概率 / Predicted probability

#### 通俗解释
多类分类就像"多选一"：不是选A或B，而是从A、B、C、D等多个选项中选一个。Softmax把所有类别的"分数"转换成概率，概率最大的就是预测类别。

---

## Additional Detail / 补充要点

### One-vs-Rest / 一对多

#### English
Train C binary classifiers, each distinguishing one class from all others. For prediction, choose the class with highest confidence.

#### 中文
训练C个二元分类器，每个区分一个类别和所有其他类别。预测时，选择置信度最高的类别。

---

### Evaluation Metrics / 评估指标

#### English
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

#### 中文
- **准确率**：(TP + TN) / (TP + TN + FP + FN)
- **精确率**：TP / (TP + FP)
- **召回率**：TP / (TP + FN)
- **F1分数**：2 × (精确率 × 召回率) / (精确率 + 召回率)

---

### Regularization in Logistic Regression / 逻辑回归中的正则化

#### English
Add L2 regularization to prevent overfitting:

$$
J(\mathbf{w}, b) = -\frac{1}{M}\sum_{i=1}^M \left[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})\right] + \frac{\lambda}{2}\|\mathbf{w}\|^2
$$

#### 中文
添加L2正则化防止过拟合：

$$
J(\mathbf{w}, b) = -\frac{1}{M}\sum_{i=1}^M \left[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})\right] + \frac{\lambda}{2}\|\mathbf{w}\|^2
$$

---

## Learning Recommendations / 学习建议

### For Beginners / 对于初学者

1. **Understand Sigmoid Function / 理解Sigmoid函数:**
   - Plot the function
   - Understand its properties
   - See how it maps to probabilities

2. **Implement Logistic Regression / 实现逻辑回归:**
   - From scratch
   - Using gradient descent
   - With regularization

3. **Visualize Decision Boundaries / 可视化决策边界:**
   - Plot data points
   - Draw decision boundary
   - See how it changes with parameters

---

## Reference Resources / 参考资源

1. **Textbooks / 教科书:**
   - Pattern Recognition and Machine Learning (Bishop)
   - Machine Learning: A Probabilistic Perspective (Murphy)

2. **Online Resources / 在线资源:**
   - Stanford CS229 Lecture Notes
   - Scikit-learn Documentation

---

## 练习题 / Practice Problems

### 问题1 / Problem 1:
推导逻辑回归的交叉熵损失函数对权重的梯度。
Derive the gradient of cross-entropy loss function with respect to weights for logistic regression.

### 问题2 / Problem 2:
解释为什么逻辑回归使用sigmoid函数而不是其他函数。
Explain why logistic regression uses sigmoid function instead of other functions.

### 问题3 / Problem 3:
对于多类分类问题，比较one-vs-rest和softmax回归两种方法。
For multiclass classification, compare one-vs-rest and softmax regression approaches.

---

## 例题与解答 / Worked Examples

### 例题1：计算Sigmoid函数值 / Calculating Sigmoid Function Value

**题目 / Question:**  
计算sigmoid函数在 $z = 0$, $z = 2$, $z = -2$ 处的值，并解释结果。
Calculate sigmoid function values at $z = 0$, $z = 2$, $z = -2$, and interpret the results.

**详细解答 / Detailed Solution:**

**Sigmoid函数定义 / Sigmoid Function Definition:**

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**计算 / Calculations:**

**1. $z = 0$:**

$$
\sigma(0) = \frac{1}{1 + e^{0}} = \frac{1}{1 + 1} = \frac{1}{2} = 0.5
$$

**2. $z = 2$:**

$$
\sigma(2) = \frac{1}{1 + e^{-2}} = \frac{1}{1 + 0.135} \approx \frac{1}{1.135} \approx 0.881
$$

**3. $z = -2$:**

$$
\sigma(-2) = \frac{1}{1 + e^{2}} = \frac{1}{1 + 7.389} \approx \frac{1}{8.389} \approx 0.119
$$

**解释 / Interpretation:**
- $z = 0$ 时，sigmoid输出 $0.5$，表示不确定 / When $z = 0$, sigmoid outputs $0.5$, indicating uncertainty
- $z > 0$ 时，sigmoid输出 $> 0.5$，倾向于正类 / When $z > 0$, sigmoid outputs $> 0.5$, favoring positive class
- $z < 0$ 时，sigmoid输出 $< 0.5$，倾向于负类 / When $z < 0$, sigmoid outputs $< 0.5$, favoring negative class
- Sigmoid将任意实数映射到 $(0,1)$ 区间，适合表示概率 / Sigmoid maps any real number to $(0,1)$ interval, suitable for representing probabilities

---

### 例题2：逻辑回归决策边界 / Logistic Regression Decision Boundary

**题目 / Question:**  
给定逻辑回归模型：$P(y=1 \mid \mathbf{x}) = \sigma(2x_1 - x_2 + 1)$，确定决策边界方程并解释其含义。
Given logistic regression model: $P(y=1 \mid \mathbf{x}) = \sigma(2x_1 - x_2 + 1)$, determine the decision boundary equation and explain its meaning.

**详细解答 / Detailed Solution:**

**步骤1：确定决策边界 / Step 1: Determine Decision Boundary**

决策边界是 $P(y=1 \mid \mathbf{x}) = 0.5$ 的点，即 / Decision boundary is where $P(y=1 \mid \mathbf{x}) = 0.5$, i.e.:
$$
\sigma(2x_1 - x_2 + 1) = 0.5
$$

由于 $\sigma(0) = 0.5$，因此 / Since $\sigma(0) = 0.5$, therefore:
$$
2x_1 - x_2 + 1 = 0
$$

**步骤2：重写为显式形式 / Step 2: Rewrite in Explicit Form**

$$
x_2 = 2x_1 + 1
$$

**步骤3：解释 / Step 3: Interpretation**

- **决策边界是一条直线** / Decision boundary is a straight line
- **在直线上方** ($x_2 > 2x_1 + 1$): $P(y=1 \mid \mathbf{x}) > 0.5$，预测类别1 / Above line: predict class 1
- **在直线下方** ($x_2 < 2x_1 + 1$): $P(y=1 \mid \mathbf{x}) < 0.5$，预测类别0 / Below line: predict class 0
- **权重 $\mathbf{w} = [2, -1]^\top$** 决定了直线的方向 / Weight $\mathbf{w} = [2, -1]^\top$ determines line direction
- **偏置 $b = 1$** 决定了直线的位置 / Bias $b = 1$ determines line position

---

