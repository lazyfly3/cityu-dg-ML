# Lecture 7 知识点总结 / Knowledge Summary

## Table of Contents / 目录

- [1. Decision Trees / 1. 决策树](#1-decision-trees--1-决策树)
- [2. Splitting Criteria / 2. 分裂准则](#2-splitting-criteria--2-分裂准则)
- [3. Pruning / 3. 剪枝](#3-pruning--3-剪枝)
- [4. Random Forests / 4. 随机森林](#4-random-forests--4-随机森林)

---

## 1. Decision Trees / 1. 决策树

#### English
Decision trees are tree-like models that make decisions by recursively splitting the data based on feature values. They are interpretable and can handle both classification and regression.

#### 中文
决策树是树状模型，通过基于特征值递归分裂数据来做决策。它们可解释，可以处理分类和回归。

**基本结构 / Basic Structure:**
- **根节点 / Root Node**: 顶层节点，包含所有数据 / Top node containing all data
- **内部节点 / Internal Node**: 决策节点，基于特征值分裂 / Decision nodes that split based on feature values
- **叶节点 / Leaf Node**: 终端节点，包含预测结果 / Terminal nodes containing predictions

**决策规则 / Decision Rules:**
每个节点根据特征值将数据分成子集 / Each node splits data into subsets based on feature values

**示例 / Example:**
```
如果 年龄 < 30:
    如果 收入 > 50000:
        预测: 购买
    否则:
        预测: 不购买
否则:
    如果 信用评分 > 700:
        预测: 购买
    否则:
        预测: 不购买
```

#### 通俗解释
决策树就像"问问题"：从根节点开始，根据特征值问"是/否"问题，一步步缩小范围，直到到达叶节点得到答案。就像玩"20个问题"游戏。

---

## 2. Splitting Criteria / 2. 分裂准则

#### English
Splitting criteria determine how to choose the best feature and threshold to split the data at each node. Common criteria include information gain, Gini impurity, and variance reduction.

#### 中文
分裂准则决定如何选择最佳特征和阈值来在每个节点分裂数据。常见准则包括信息增益、基尼不纯度和方差减少。

**信息增益 / Information Gain:**

**熵 / Entropy:**
$$
H(S) = -\sum_{i=1}^C p_i \log_2 p_i
$$

**信息增益 / Information Gain:**
$$
IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)
$$

其中 / where:
- $S$：当前数据集 / Current dataset
- $A$：特征 / Feature
- $S_v$：特征A取值为v的子集 / Subset where feature A has value v

**基尼不纯度 / Gini Impurity:**
$$
Gini(S) = 1 - \sum_{i=1}^C p_i^2
$$

**基尼增益 / Gini Gain:**
$$
GiniGain(S, A) = Gini(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} Gini(S_v)
$$

**方差减少（回归）/ Variance Reduction (Regression):**
$$
\text{Var}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y})^2
$$

$$
\text{VarReduction}(S, A) = \text{Var}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Var}(S_v)
$$

**选择最佳分裂 / Choosing Best Split:**
选择使增益最大的特征和阈值 / Choose feature and threshold that maximize gain

#### 通俗解释
分裂准则就像"找最好的问题"：在每一步，我们找一个问题（特征），问这个问题后能让数据"更纯"（同类数据更集中）。信息增益衡量"问这个问题能减少多少不确定性"。

---

## 3. Pruning / 3. 剪枝

#### English
Pruning is a technique to reduce overfitting by removing branches that don't contribute significantly to the model's performance. It simplifies the tree and improves generalization.

#### 中文
剪枝是一种通过移除对模型性能贡献不大的分支来减少过拟合的技术。它简化树并提高泛化能力。

**预剪枝 / Pre-pruning (Early Stopping):**
在构建树时停止分裂 / Stop splitting during tree construction

**停止条件 / Stopping Criteria:**
- 节点中样本数少于阈值 / Number of samples in node below threshold
- 信息增益小于阈值 / Information gain below threshold
- 达到最大深度 / Maximum depth reached
- 所有样本属于同一类 / All samples belong to same class

**后剪枝 / Post-pruning:**
先构建完整树，然后移除分支 / Build full tree first, then remove branches

**方法 / Methods:**
- **成本复杂度剪枝 / Cost-Complexity Pruning**: 平衡树复杂度和误差
- **减少误差剪枝 / Reduced Error Pruning**: 使用验证集评估

**成本复杂度 / Cost-Complexity:**
$$
C_\alpha(T) = \sum_{t \in \text{leaves}} N_t H(t) + \alpha |T|
$$

其中 / where:
- $N_t$：叶节点t的样本数 / Number of samples in leaf t
- $H(t)$：叶节点t的熵 / Entropy of leaf t
- $|T|$：树的叶节点数 / Number of leaf nodes
- $\alpha$：复杂度参数 / Complexity parameter

#### 通俗解释
剪枝就像"修剪树枝"：树长得太复杂（过拟合），我们剪掉一些不重要的分支，让树更简单、更稳健。预剪枝是"长的时候控制"，后剪枝是"长完了再修剪"。

---

## 4. Random Forests / 4. 随机森林

#### English
Random Forests are ensemble methods that combine multiple decision trees. Each tree is trained on a random subset of data and features, and predictions are made by voting (classification) or averaging (regression).

#### 中文
随机森林是组合多个决策树的集成方法。每棵树在数据的随机子集和特征的随机子集上训练，通过投票（分类）或平均（回归）进行预测。

**算法 / Algorithm:**

1. **Bootstrap采样 / Bootstrap Sampling:**
   从训练集中有放回地随机抽取M个样本 / Randomly sample M samples with replacement from training set

2. **特征子集选择 / Feature Subset Selection:**
   对每个节点，随机选择$\sqrt{d}$个特征（d是总特征数）/ For each node, randomly select $\sqrt{d}$ features (d is total number of features)

3. **构建树 / Build Tree:**
   使用选定的样本和特征构建决策树 / Build decision tree using selected samples and features

4. **重复 / Repeat:**
   重复步骤1-3构建多棵树 / Repeat steps 1-3 to build multiple trees

5. **预测 / Prediction:**
   - 分类：多数投票 / Classification: majority voting
   - 回归：平均 / Regression: average

**优点 / Advantages:**
- 减少过拟合 / Reduces overfitting
- 处理高维数据 / Handles high-dimensional data
- 提供特征重要性 / Provides feature importance
- 对缺失值鲁棒 / Robust to missing values

**特征重要性 / Feature Importance:**
通过计算每个特征在所有树中的平均信息增益来评估 / Evaluate by computing average information gain of each feature across all trees

#### 通俗解释
随机森林就像"多个专家投票"：训练很多棵不同的树（每个专家），每棵树看不同的数据和特征，最后所有专家投票决定答案。这样比单个专家更可靠，不容易出错。

---

## Additional Detail / 补充要点

### Handling Continuous Features / 处理连续特征

#### English
For continuous features, find the best threshold to split:
1. Sort feature values
2. Consider midpoints between consecutive values as candidate thresholds
3. Choose threshold that maximizes splitting criterion

#### 中文
对于连续特征，找到最佳分裂阈值：
1. 对特征值排序
2. 考虑连续值之间的中点作为候选阈值
3. 选择使分裂准则最大的阈值

---

### Handling Missing Values / 处理缺失值

#### English
- **Surrogate splits**: Use other features when primary feature is missing
- **Default direction**: Send missing values to most common branch
- **Imputation**: Fill missing values before training

#### 中文
- **代理分裂**：当主特征缺失时使用其他特征
- **默认方向**：将缺失值发送到最常见的分支
- **插补**：在训练前填充缺失值

---

### Advantages and Disadvantages / 优缺点

#### English
**Advantages:**
- Interpretable
- Handles non-linear relationships
- No feature scaling needed
- Handles mixed data types

**Disadvantages:**
- Can overfit without pruning
- Unstable (small data changes can change tree structure)
- May not capture additive relationships well

#### 中文
**优点:**
- 可解释
- 处理非线性关系
- 不需要特征缩放
- 处理混合数据类型

**缺点:**
- 不剪枝可能过拟合
- 不稳定（小的数据变化可能改变树结构）
- 可能不能很好地捕捉加性关系

---

## Learning Recommendations / 学习建议

### For Beginners / 对于初学者

1. **Build a Simple Tree / 构建简单树:**
   - Implement from scratch
   - Understand splitting criteria
   - Visualize the tree

2. **Experiment with Pruning / 实验剪枝:**
   - Compare pruned vs unpruned trees
   - See effect on overfitting
   - Tune hyperparameters

3. **Implement Random Forest / 实现随机森林:**
   - Understand bootstrap sampling
   - See how ensemble improves performance
   - Compare with single tree

---

## Reference Resources / 参考资源

1. **Textbooks / 教科书:**
   - The Elements of Statistical Learning (Hastie et al.)
   - Pattern Recognition and Machine Learning (Bishop)

2. **Online Resources / 在线资源:**
   - Scikit-learn Decision Trees Documentation
   - Random Forest Paper (Breiman, 2001)

---

## 练习题 / Practice Problems

### 问题1 / Problem 1:
解释信息增益和基尼不纯度的区别，并说明何时使用哪个。
Explain the difference between information gain and Gini impurity, and explain when to use which.

### 问题2 / Problem 2:
比较预剪枝和后剪枝的优缺点。
Compare the advantages and disadvantages of pre-pruning and post-pruning.

### 问题3 / Problem 3:
解释随机森林如何减少过拟合，并说明为什么需要随机选择特征。
Explain how random forests reduce overfitting and explain why random feature selection is needed.

---

## 例题与解答 / Worked Examples

### 例题1：计算信息增益 / Calculating Information Gain

**题目 / Question:**  
给定数据集，计算按特征A分裂的信息增益。
Given dataset, calculate information gain for splitting by feature A.

数据集 / Dataset:
- 类别分布：5个正例，5个负例 / Class distribution: 5 positive, 5 negative
- 按特征A分裂后：左分支3正1负，右分支2正4负 / After splitting by feature A: left branch 3 positive 1 negative, right branch 2 positive 4 negative

**详细解答 / Detailed Solution:**

**步骤1：计算父节点的熵 / Step 1: Calculate Parent Node Entropy**
$$
H(S) = -\frac{5}{10}\log_2\frac{5}{10} - \frac{5}{10}\log_2\frac{5}{10} = -0.5 \times (-1) - 0.5 \times (-1) = 1.0
$$

**步骤2：计算子节点的熵 / Step 2: Calculate Child Node Entropies**

左分支 / Left branch:
$$
H(S_{\text{left}}) = -\frac{3}{4}\log_2\frac{3}{4} - \frac{1}{4}\log_2\frac{1}{4} \approx 0.811
$$

右分支 / Right branch:
$$
H(S_{\text{right}}) = -\frac{2}{6}\log_2\frac{2}{6} - \frac{4}{6}\log_2\frac{4}{6} \approx 0.918
$$

**步骤3：计算加权平均熵 / Step 3: Calculate Weighted Average Entropy**
$$
H(S|A) = \frac{4}{10} \times 0.811 + \frac{6}{10} \times 0.918 \approx 0.875
$$

**步骤4：计算信息增益 / Step 4: Calculate Information Gain**
$$
IG(S, A) = H(S) - H(S|A) = 1.0 - 0.875 = 0.125
$$

**结论 / Conclusion:**
按特征 $A$ 分裂的信息增益为 $0.125$。
Information gain for splitting by feature $A$ is $0.125$.

---

### 例题2：随机森林预测 / Random Forest Prediction

**题目 / Question:**  
解释随机森林如何对新的测试样本进行预测。
Explain how random forest makes predictions for a new test sample.

**详细解答 / Detailed Solution:**

**随机森林预测过程 / Random Forest Prediction Process:**

1. **输入新样本 / Input New Sample:**
   给定特征向量$\mathbf{x}$ / Given feature vector $\mathbf{x}$

2. **每个树进行预测 / Each Tree Makes Prediction:**
   - 树1预测：类别A / Tree 1 predicts: class A
   - 树2预测：类别A / Tree 2 predicts: class A
   - 树3预测：类别B / Tree 3 predicts: class B
   - 树4预测：类别A / Tree 4 predicts: class A
   - 树5预测：类别A / Tree 5 predicts: class A

3. **投票决定 / Voting Decision:**
   - 类别A：4票 / Class A: 4 votes
   - 类别B：1票 / Class B: 1 vote
   - 多数投票：类别A / Majority vote: class A

4. **最终预测 / Final Prediction:**
   随机森林预测类别A / Random forest predicts class A

**优点 / Advantages:**
- **鲁棒性 / Robustness**: 即使少数树预测错误，多数投票仍能给出正确结果 / Even if few trees predict wrong, majority vote can still give correct result
- **减少方差 / Variance Reduction**: 多个模型的平均减少预测方差 / Average of multiple models reduces prediction variance
- **处理噪声 / Handle Noise**: 对噪声数据更鲁棒 / More robust to noisy data

**结论 / Conclusion:**
随机森林通过集成多个决策树的预测，使用多数投票或平均来做出最终预测，通常比单个决策树更准确和稳定。
Random forest makes final prediction by ensemble of multiple decision trees using majority voting or averaging, usually more accurate and stable than single decision tree.

---

