# Lecture 10 知识点总结 / Knowledge Summary

## Table of Contents / 目录

- [1. Clustering / 1. 聚类](#1-clustering--1-聚类)
- [2. K-Means / 2. K均值](#2-k-means--2-k均值)
- [3. Hierarchical Clustering / 3. 层次聚类](#3-hierarchical-clustering--3-层次聚类)
- [4. Evaluation Metrics / 4. 评估指标](#4-evaluation-metrics--4-评估指标)

---

## 1. Clustering / 1. 聚类

#### English
Clustering is an unsupervised learning task that groups similar data points together without using labeled data. The goal is to discover hidden patterns and structures in the data.

#### 中文
聚类是一种无监督学习任务，在不使用标签数据的情况下将相似的数据点分组。目标是发现数据中隐藏的模式和结构。

**基本概念 / Basic Concepts:**
- **簇（Cluster）**: 一组相似的数据点 / Group of similar data points
- **质心（Centroid）**: 簇的中心点 / Center point of cluster
- **距离度量（Distance Metric）**: 衡量数据点相似性的方法 / Method to measure similarity between data points

**应用 / Applications:**
- 客户细分 / Customer segmentation
- 图像分割 / Image segmentation
- 异常检测 / Anomaly detection
- 数据压缩 / Data compression

#### 通俗解释
聚类就像"物以类聚"：把相似的东西放在一起，不相似的分开。就像整理房间：把书放一起，衣服放一起，不需要标签，自动分类。

---

## 2. K-Means / 2. K均值

#### English
K-Means is a popular clustering algorithm that partitions data into K clusters by minimizing the within-cluster sum of squares. It iteratively assigns points to nearest centroids and updates centroids.

#### 中文
K-Means是一种流行的聚类算法，通过最小化簇内平方和将数据划分为K个簇。它迭代地将点分配给最近的质心并更新质心。

**算法 / Algorithm:**

1. **初始化 / Initialization:**
   随机选择K个初始质心 / Randomly select K initial centroids

2. **分配 / Assignment:**
   将每个数据点分配给最近的质心 / Assign each data point to nearest centroid:
   $$
   c^{(i)} = \arg\min_{k} \|\mathbf{x}^{(i)} - \boldsymbol{\mu}_k\|
   $$

3. **更新 / Update:**
   重新计算每个簇的质心 / Recompute centroid for each cluster:
   $$
   \boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{i \in C_k} \mathbf{x}^{(i)}
   $$

4. **重复 / Repeat:**
   重复步骤2-3直到收敛 / Repeat steps 2-3 until convergence

**目标函数 / Objective Function:**
$$
J = \sum_{i=1}^M \sum_{k=1}^K r_{ik} \|\mathbf{x}^{(i)} - \boldsymbol{\mu}_k\|^2
$$

其中 / where:
- $r_{ik} = 1$如果$\mathbf{x}^{(i)}$属于簇$k$，否则为0 / $r_{ik} = 1$ if $\mathbf{x}^{(i)}$ belongs to cluster $k$, else 0

**初始化方法 / Initialization Methods:**
- **随机初始化 / Random**: 随机选择K个点 / Randomly select K points
- **K-Means++**: 选择相距较远的初始点 / Select initial points far apart

**计算步骤 / Calculation Steps:**
1. 选择K值 / Choose K value
2. 初始化K个质心 / Initialize K centroids
3. 对每个数据点，计算到所有质心的距离 / For each data point, compute distances to all centroids
4. 将点分配给最近的质心 / Assign points to nearest centroids
5. 更新质心位置 / Update centroid positions
6. 检查是否收敛（质心不再变化或变化很小）/ Check convergence
7. 如果未收敛，返回步骤3 / If not converged, return to step 3

#### 通俗解释
K-Means就像"找中心"：先随机选K个点作为"中心"，然后把所有点分配给最近的中心，再重新计算每个组的"中心"，重复直到中心不再变化。就像选班长：先选几个候选人，大家投票给最近的，然后重新选中心，直到稳定。

---

## 3. Hierarchical Clustering / 3. 层次聚类

#### English
Hierarchical clustering builds a tree of clusters (dendrogram) by either merging small clusters (agglomerative) or splitting large clusters (divisive). It doesn't require specifying the number of clusters in advance.

#### 中文
层次聚类通过合并小簇（凝聚式）或分裂大簇（分裂式）构建簇的树（树状图）。它不需要预先指定簇的数量。

**凝聚式层次聚类 / Agglomerative Hierarchical Clustering:**

**算法 / Algorithm:**
1. 开始时每个点是一个簇 / Start with each point as a cluster
2. 重复直到只剩一个簇 / Repeat until one cluster remains:
   - 找到最近的两个簇 / Find two closest clusters
   - 合并这两个簇 / Merge these two clusters
   - 更新距离矩阵 / Update distance matrix

**距离度量 / Distance Metrics:**

1. **单链接（Single Linkage）**: 最近点距离 / Nearest point distance
   $$
   d(C_i, C_j) = \min_{\mathbf{x} \in C_i, \mathbf{y} \in C_j} \|\mathbf{x} - \mathbf{y}\|
   $$

2. **全链接（Complete Linkage）**: 最远点距离 / Farthest point distance
   $$
   d(C_i, C_j) = \max_{\mathbf{x} \in C_i, \mathbf{y} \in C_j} \|\mathbf{x} - \mathbf{y}\|
   $$

3. **平均链接（Average Linkage）**: 平均距离 / Average distance
   $$
   d(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{\mathbf{x} \in C_i} \sum_{\mathbf{y} \in C_j} \|\mathbf{x} - \mathbf{y}\|
   $$

4. **质心链接（Centroid Linkage）**: 质心距离 / Centroid distance
   $$
   d(C_i, C_j) = \|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\|
   $$

**树状图 / Dendrogram:**
可视化层次聚类结果 / Visualize hierarchical clustering results
- 高度表示合并距离 / Height represents merge distance
- 切割高度决定簇数量 / Cut height determines number of clusters

#### 通俗解释
层次聚类就像"家族树"：从每个人（点）开始，逐步合并成家庭、家族，最终成为一个大族。树状图显示合并过程，可以在任意高度"切一刀"得到不同数量的簇。

---

## 4. Evaluation Metrics / 4. 评估指标

#### English
Clustering evaluation metrics measure the quality of clusters. They can be internal (based on data only) or external (using ground truth labels).

#### 中文
聚类评估指标衡量簇的质量。它们可以是内部的（仅基于数据）或外部的（使用真实标签）。

**内部指标 / Internal Metrics:**

1. **轮廓系数（Silhouette Coefficient）:**
   $$
   s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
   $$
   其中 / where:
   - $a(i)$：点$i$到同簇其他点的平均距离 / Average distance from point $i$ to other points in same cluster
   - $b(i)$：点$i$到最近其他簇的平均距离 / Average distance from point $i$ to nearest other cluster
   - 范围：[-1, 1]，越大越好 / Range: [-1, 1], larger is better

2. **簇内平方和（Within-Cluster Sum of Squares, WCSS）:**
   $$
   WCSS = \sum_{k=1}^K \sum_{i \in C_k} \|\mathbf{x}^{(i)} - \boldsymbol{\mu}_k\|^2
   $$
   越小越好 / Smaller is better

3. **簇间平方和（Between-Cluster Sum of Squares, BCSS）:**
   $$
   BCSS = \sum_{k=1}^K |C_k| \|\boldsymbol{\mu}_k - \boldsymbol{\mu}\|^2
   $$
   越大越好 / Larger is better

**外部指标 / External Metrics:**

1. **调整兰德指数（Adjusted Rand Index, ARI）:**
   $$
   ARI = \frac{\sum_{ij} \binom{n_{ij}}{2} - [\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}] / \binom{n}{2}}{\frac{1}{2}[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}] - [\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}] / \binom{n}{2}}
   $$
   范围：[-1, 1]，越大越好 / Range: [-1, 1], larger is better

2. **归一化互信息（Normalized Mutual Information, NMI）:**
   $$
   NMI = \frac{2 \cdot I(C; K)}{H(C) + H(K)}
   $$
   范围：[0, 1]，越大越好 / Range: [0, 1], larger is better

#### 通俗解释
评估指标就像"打分"：
- **内部指标**：看簇是否"紧密"（簇内近）和"分离"（簇间远）
- **外部指标**：如果有正确答案，看预测和答案有多像

---

## Additional Detail / 补充要点

### Choosing K in K-Means / 选择K值

#### English
- **Elbow Method**: Plot WCSS vs K, look for "elbow"
- **Silhouette Analysis**: Choose K that maximizes average silhouette score
- **Domain Knowledge**: Use prior knowledge about data

#### 中文
- **肘部方法**：绘制WCSS vs K，寻找"肘部"
- **轮廓分析**：选择使平均轮廓分数最大的K
- **领域知识**：使用关于数据的先验知识

---

### DBSCAN / DBSCAN算法

#### English
Density-Based Spatial Clustering of Applications with Noise (DBSCAN) finds clusters of arbitrary shape based on density. It doesn't require specifying K.

#### 中文
基于密度的噪声应用空间聚类（DBSCAN）基于密度找到任意形状的簇。它不需要指定K。

**参数 / Parameters:**
- **eps**: 邻域半径 / Neighborhood radius
- **minPts**: 形成簇所需的最小点数 / Minimum points to form cluster

---

### Advantages and Disadvantages / 优缺点

#### English
**K-Means:**
- **Pros**: Simple, fast, works well for spherical clusters
- **Cons**: Requires K, sensitive to initialization, assumes spherical clusters

**Hierarchical:**
- **Pros**: No need to specify K, produces dendrogram
- **Cons**: Computationally expensive, sensitive to noise

#### 中文
**K-Means:**
- **优点**: 简单、快速，对球形簇效果好
- **缺点**: 需要K，对初始化敏感，假设球形簇

**层次聚类:**
- **优点**: 不需要指定K，产生树状图
- **缺点**: 计算昂贵，对噪声敏感

---

## Learning Recommendations / 学习建议

### For Beginners / 对于初学者

1. **Implement K-Means / 实现K-Means:**
   - From scratch
   - Understand initialization
   - Visualize iterations

2. **Experiment with Different K / 实验不同K值:**
   - Use elbow method
   - Compute silhouette scores
   - See how clusters change

3. **Compare Algorithms / 比较算法:**
   - K-Means vs Hierarchical
   - Different distance metrics
   - On different datasets

---

## Reference Resources / 参考资源

1. **Textbooks / 教科书:**
   - Pattern Recognition and Machine Learning (Bishop)
   - The Elements of Statistical Learning (Hastie et al.)

2. **Online Resources / 在线资源:**
   - Scikit-learn Clustering Documentation
   - Stanford CS229 Lecture Notes

---

## 练习题 / Practice Problems

### 问题1 / Problem 1:
解释K-Means算法的收敛性，并说明算法何时停止。
Explain the convergence of K-Means algorithm and explain when the algorithm stops.

### 问题2 / Problem 2:
比较K-Means和层次聚类的优缺点，并说明各自的适用场景。
Compare the advantages and disadvantages of K-Means and hierarchical clustering, and explain their respective use cases.

### 问题3 / Problem 3:
解释如何选择K-Means中的K值，并比较不同的方法。
Explain how to choose K value in K-Means and compare different methods.

---

## 例题与解答 / Worked Examples

### 例题1：K-Means迭代计算 / K-Means Iteration Calculation

**题目 / Question:**  
给定数据点：(1, 1), (1, 2), (2, 1), (8, 8), (9, 8)，使用K=2进行K-Means聚类，初始质心为(1, 1)和(8, 8)。
Given data points: (1, 1), (1, 2), (2, 1), (8, 8), (9, 8), use K=2 for K-Means clustering with initial centroids (1, 1) and (8, 8).

**详细解答 / Detailed Solution:**

**步骤1：第一次迭代 - 分配 / Step 1: First Iteration - Assignment**

计算每个点到质心的距离 / Calculate distance from each point to centroids:

点到(1,1)的距离 / Distance to (1,1):
- (1,1): 0
- (1,2): 1
- (2,1): 1
- (8,8): 9.90
- (9,8): 10.63

点到(8,8)的距离 / Distance to (8,8):
- (1,1): 9.90
- (1,2): 9.22
- (2,1): 9.22
- (8,8): 0
- (9,8): 1

分配结果 / Assignment:
- 簇1：(1,1), (1,2), (2,1) / Cluster 1: (1,1), (1,2), (2,1)
- 簇2：(8,8), (9,8) / Cluster 2: (8,8), (9,8)

**步骤2：第一次迭代 - 更新质心 / Step 2: First Iteration - Update Centroids**

新质心 / New centroids:
- 簇1质心 / Cluster 1 centroid: ((1+1+2)/3, (1+2+1)/3) = (1.33, 1.33)
- 簇2质心 / Cluster 2 centroid: ((8+9)/2, (8+8)/2) = (8.5, 8.0)

**步骤3：第二次迭代 / Step 3: Second Iteration**

重新分配后，质心不再变化，算法收敛 / After reassignment, centroids no longer change, algorithm converges.

**结论 / Conclusion:**
K-Means在两次迭代后收敛，最终簇1包含前3个点，簇2包含后2个点。
K-Means converges after two iterations, final cluster 1 contains first 3 points, cluster 2 contains last 2 points.

---

### 例题2：选择K值 - 肘部方法 / Choosing K - Elbow Method

**题目 / Question:**  
解释如何使用肘部方法选择K-Means中的K值。
Explain how to use elbow method to choose K value in K-Means.

**详细解答 / Detailed Solution:**

**肘部方法步骤 / Elbow Method Steps:**

1. **尝试不同的K值 / Try Different K Values:**
   对K = 1, 2, 3, ..., K_max运行K-Means / Run K-Means for K = 1, 2, 3, ..., K_max

2. **计算WCSS / Calculate WCSS:**
   对每个K，计算簇内平方和 / For each K, calculate within-cluster sum of squares:
   $$
   WCSS(K) = \sum_{k=1}^K \sum_{i \in C_k} \|\mathbf{x}^{(i)} - \boldsymbol{\mu}_k\|^2
   $$

3. **绘制WCSS vs K / Plot WCSS vs K:**
   绘制曲线，寻找"肘部"（曲线急剧下降后变平缓的点）/ Plot curve, look for "elbow" (point where curve drops sharply then flattens)

**示例数据 / Example Data:**
假设WCSS值如下 / Suppose WCSS values:
- K=1: WCSS = 100
- K=2: WCSS = 30
- K=3: WCSS = 15
- K=4: WCSS = 10
- K=5: WCSS = 8
- K=6: WCSS = 7

**分析 / Analysis:**
- K=1到K=2：WCSS大幅下降（100→30）/ K=1 to K=2: large WCSS drop (100→30)
- K=2到K=3：WCSS继续下降（30→15）/ K=2 to K=3: continued WCSS drop (30→15)
- K=3到K=4：WCSS下降较小（15→10）/ K=3 to K=4: smaller WCSS drop (15→10)
- K=4之后：WCSS下降很小 / After K=4: very small WCSS drops

**结论 / Conclusion:**
肘部在K=3或K=4处，选择K=3或K=4。通常选择K=3，因为增加K=4带来的改善很小。
Elbow is at K=3 or K=4, choose K=3 or K=4. Usually choose K=3, as improvement from K=4 is small.

---

