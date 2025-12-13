# Lecture 11 知识点总结 / Knowledge Summary

## Table of Contents / 目录

- [1. Ensemble Methods / 1. 集成方法](#1-ensemble-methods--1-集成方法)
- [2. Bagging / 2. 装袋法](#2-bagging--2-装袋法)
- [3. Boosting / 3. 提升法](#3-boosting--3-提升法)
- [4. Stacking / 4. 堆叠法](#4-stacking--4-堆叠法)

---

## 1. Ensemble Methods / 1. 集成方法

#### English
Ensemble methods combine multiple base learners to create a more powerful model. They typically achieve better performance than individual models by reducing variance, bias, or both.

#### 中文
集成方法组合多个基学习器以创建更强大的模型。它们通常通过减少方差、偏差或两者来实现比单个模型更好的性能。

**基本思想 / Basic Idea:**
- **多样性 / Diversity**: 基学习器应该不同 / Base learners should be different
- **组合策略 / Combination Strategy**: 如何组合预测 / How to combine predictions
- **性能提升 / Performance Improvement**: 集成通常比单个模型更好 / Ensemble usually better than individual models

**组合方法 / Combination Methods:**
- **投票（Voting）**: 多数投票或加权投票 / Majority voting or weighted voting
- **平均（Averaging）**: 预测值的平均 / Average of predictions
- **学习组合（Learned Combination）**: 训练元学习器组合预测 / Train meta-learner to combine predictions

**理论依据 / Theoretical Justification:**
- **偏差-方差分解 / Bias-Variance Decomposition**: 集成可以减少方差 / Ensemble can reduce variance
- **大数定律 / Law of Large Numbers**: 多个模型的平均更稳定 / Average of many models is more stable

#### 通俗解释
集成方法就像"三个臭皮匠顶个诸葛亮"：多个模型一起做决策，比单个模型更可靠。就像考试时多个同学讨论答案，比一个人想更准确。

---

## 2. Bagging / 2. 装袋法

#### English
Bagging (Bootstrap Aggregating) trains multiple models on different bootstrap samples of the training data and combines their predictions by voting or averaging.

#### 中文
装袋法（Bootstrap聚合）在训练数据的不同Bootstrap样本上训练多个模型，并通过投票或平均组合它们的预测。

**算法 / Algorithm:**

1. **Bootstrap采样 / Bootstrap Sampling:**
   从训练集中有放回地随机抽取$M$个样本 / Randomly sample $M$ samples with replacement from training set
   - 每个Bootstrap样本大小等于原始训练集 / Each bootstrap sample same size as original training set
   - 有些样本可能被选中多次，有些可能不被选中 / Some samples may be selected multiple times, some may not be selected

2. **训练基学习器 / Train Base Learners:**
   在每个Bootstrap样本上训练一个模型 / Train one model on each bootstrap sample

3. **组合预测 / Combine Predictions:**
   - **分类 / Classification**: 多数投票 / Majority voting
     $$
     \hat{y} = \arg\max_{y} \sum_{i=1}^T \mathbf{1}[\hat{y}_i = y]
     $$
   - **回归 / Regression**: 平均 / Average
     $$
     \hat{y} = \frac{1}{T}\sum_{i=1}^T \hat{y}_i
     $$

**随机森林（Random Forest） / Random Forest:**
- Bagging + 决策树 / Bagging + Decision Trees
- 每个节点随机选择特征子集 / Random feature subset at each node
- 进一步增加多样性 / Further increases diversity

**优点 / Advantages:**
- 减少方差 / Reduces variance
- 减少过拟合 / Reduces overfitting
- 可以并行训练 / Can train in parallel
- 提供特征重要性 / Provides feature importance

**计算步骤 / Calculation Steps:**
1. 选择基学习器类型（如决策树）/ Choose base learner type (e.g., decision tree)
2. 选择集成大小$T$（模型数量）/ Choose ensemble size $T$ (number of models)
3. 对$t = 1, \ldots, T$:
   - 从训练集Bootstrap采样 / Bootstrap sample from training set
   - 在Bootstrap样本上训练模型$h_t$ / Train model $h_t$ on bootstrap sample
4. 组合所有模型的预测 / Combine predictions from all models

#### 通俗解释
Bagging就像"多个专家独立判断"：每个专家看不同的数据（Bootstrap采样），独立做判断，最后投票决定。随机森林是"多个决策树专家"投票，每个专家看不同的数据和特征。

---

## 3. Boosting / 3. 提升法

#### English
Boosting trains models sequentially, where each new model focuses on examples that previous models got wrong. It combines weak learners into a strong learner.

#### 中文
提升法顺序训练模型，每个新模型专注于之前模型出错的样本。它将弱学习器组合成强学习器。

**AdaBoost算法 / AdaBoost Algorithm:**

1. **初始化 / Initialization:**
   初始化样本权重：$w_i^{(1)} = \frac{1}{M}$ / Initialize sample weights: $w_i^{(1)} = \frac{1}{M}$

2. **对$t = 1, \ldots, T$ / For $t = 1, \ldots, T$:**
   - 在加权数据上训练弱学习器$h_t$ / Train weak learner $h_t$ on weighted data
   - 计算加权误差 / Compute weighted error:
     $$
     \epsilon_t = \sum_{i=1}^M w_i^{(t)} \mathbf{1}[h_t(\mathbf{x}^{(i)}) \ne y^{(i)}]
     $$
   - 计算学习器权重 / Compute learner weight:
     $$
     \alpha_t = \frac{1}{2}\log\frac{1-\epsilon_t}{\epsilon_t}
     $$
   - 更新样本权重 / Update sample weights:
     $$
     w_i^{(t+1)} = \frac{w_i^{(t)}}{Z_t} \exp(-\alpha_t y^{(i)} h_t(\mathbf{x}^{(i)}))
     $$
     其中$Z_t$是归一化常数 / where $Z_t$ is normalization constant

3. **最终预测 / Final Prediction:**
   $$
   H(\mathbf{x}) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(\mathbf{x})\right)
   $$

**梯度提升（Gradient Boosting） / Gradient Boosting:**
- 使用梯度下降优化损失函数 / Use gradient descent to optimize loss function
- 每个新模型拟合前一个模型的残差 / Each new model fits residuals of previous model

**XGBoost / XGBoost:**
- 梯度提升的高效实现 / Efficient implementation of gradient boosting
- 使用二阶梯度信息 / Uses second-order gradient information
- 正则化防止过拟合 / Regularization to prevent overfitting

#### 通俗解释
Boosting就像"从错误中学习"：第一个模型可能有些样本分错，第二个模型重点关注这些错分的样本，第三个模型关注前两个都错的样本，这样一步步改进。AdaBoost是"加权投票"：表现好的模型权重高，表现差的权重低。

---

## 4. Stacking / 4. 堆叠法

#### English
Stacking trains a meta-learner to combine predictions from multiple base learners. It uses cross-validation to generate training data for the meta-learner.

#### 中文
堆叠法训练一个元学习器来组合多个基学习器的预测。它使用交叉验证为元学习器生成训练数据。

**算法 / Algorithm:**

1. **训练基学习器 / Train Base Learners:**
   使用K折交叉验证训练每个基学习器 / Train each base learner using K-fold cross-validation

2. **生成元特征 / Generate Meta-Features:**
   对每个样本，使用不在其所在折的模型预测 / For each sample, use predictions from models not in its fold
   - 避免数据泄漏 / Avoid data leakage
   - 元特征：基学习器的预测 / Meta-features: predictions from base learners

3. **训练元学习器 / Train Meta-Learner:**
   在元特征上训练元学习器 / Train meta-learner on meta-features
   - 输入：基学习器的预测 / Input: predictions from base learners
   - 输出：最终预测 / Output: final prediction

4. **最终预测 / Final Prediction:**
   使用所有基学习器和元学习器进行预测 / Use all base learners and meta-learner for prediction

**示例 / Example:**
- 基学习器：决策树、SVM、逻辑回归 / Base learners: Decision Tree, SVM, Logistic Regression
- 元学习器：线性回归或神经网络 / Meta-learner: Linear Regression or Neural Network

#### 通俗解释
Stacking就像"让专家选专家"：多个基学习器（专家）先做预测，然后元学习器（总指挥）学习如何组合这些专家的意见。就像让多个医生诊断，然后让主任医师综合所有意见做最终诊断。

---

## Additional Detail / 补充要点

### Bias-Variance Tradeoff / 偏差-方差权衡

#### English
- **Bagging**: Primarily reduces variance
- **Boosting**: Reduces both bias and variance
- **Stacking**: Can reduce both depending on meta-learner

#### 中文
- **Bagging**: 主要减少方差
- **Boosting**: 同时减少偏差和方差
- **Stacking**: 根据元学习器可以减少两者

---

### When to Use Each / 何时使用每种方法

#### English
- **Bagging**: When base learners have high variance (e.g., deep trees)
- **Boosting**: When base learners are weak (e.g., shallow trees)
- **Stacking**: When you have diverse base learners and want optimal combination

#### 中文
- **Bagging**: 当基学习器方差高时（如深树）
- **Boosting**: 当基学习器弱时（如浅树）
- **Stacking**: 当有多样化的基学习器且想要最优组合时

---

### Computational Considerations / 计算考虑

#### English
- **Bagging**: Can be parallelized easily
- **Boosting**: Sequential, harder to parallelize
- **Stacking**: Requires training multiple models and meta-learner

#### 中文
- **Bagging**: 易于并行化
- **Boosting**: 顺序的，难以并行化
- **Stacking**: 需要训练多个模型和元学习器

---

## Learning Recommendations / 学习建议

### For Beginners / 对于初学者

1. **Understand the Differences / 理解差异:**
   - Bagging: Independent models
   - Boosting: Sequential models
   - Stacking: Learned combination

2. **Implement Simple Versions / 实现简单版本:**
   - Bagging with decision trees
   - AdaBoost
   - Simple stacking

3. **Experiment on Datasets / 在数据集上实验:**
   - Compare individual vs ensemble
   - Compare different ensemble methods
   - Tune hyperparameters

---

## Reference Resources / 参考资源

1. **Textbooks / 教科书:**
   - The Elements of Statistical Learning (Hastie et al.)
   - Pattern Recognition and Machine Learning (Bishop)

2. **Online Resources / 在线资源:**
   - Scikit-learn Ensemble Documentation
   - XGBoost Documentation

---

## 练习题 / Practice Problems

### 问题1 / Problem 1:
解释Bagging如何减少方差，并说明为什么需要Bootstrap采样。
Explain how Bagging reduces variance and explain why Bootstrap sampling is needed.

### 问题2 / Problem 2:
比较AdaBoost和梯度提升的区别，并说明各自的优缺点。
Compare the differences between AdaBoost and gradient boosting, and explain their respective advantages and disadvantages.

### 问题3 / Problem 3:
解释Stacking中为什么需要使用交叉验证来生成元特征。
Explain why cross-validation is needed to generate meta-features in Stacking.

---

## 例题与解答 / Worked Examples

### 例题1：Bagging预测 / Bagging Prediction

**题目 / Question:**  
解释Bagging如何对新的测试样本进行预测，并说明为什么集成预测通常比单个模型更好。
Explain how Bagging makes predictions for a new test sample and explain why ensemble predictions are usually better than individual models.

**详细解答 / Detailed Solution:**

**Bagging预测过程 / Bagging Prediction Process:**

1. **训练多个模型 / Train Multiple Models:**
   - 模型1：在Bootstrap样本1上训练 / Model 1: trained on bootstrap sample 1
   - 模型2：在Bootstrap样本2上训练 / Model 2: trained on bootstrap sample 2
   - ...
   - 模型T：在Bootstrap样本T上训练 / Model T: trained on bootstrap sample T

2. **每个模型进行预测 / Each Model Makes Prediction:**
   - 模型1预测：$\hat{y}_1 = 0.7$ / Model 1 predicts: $\hat{y}_1 = 0.7$
   - 模型2预测：$\hat{y}_2 = 0.8$ / Model 2 predicts: $\hat{y}_2 = 0.8$
   - 模型3预测：$\hat{y}_3 = 0.6$ / Model 3 predicts: $\hat{y}_3 = 0.6$
   - 模型4预测：$\hat{y}_4 = 0.75$ / Model 4 predicts: $\hat{y}_4 = 0.75$
   - 模型5预测：$\hat{y}_5 = 0.65$ / Model 5 predicts: $\hat{y}_5 = 0.65$

3. **组合预测 / Combine Predictions:**
   - **分类 / Classification**: 多数投票 / Majority voting
   - **回归 / Regression**: 平均 / Average
     $$
     \hat{y} = \frac{1}{5}(0.7 + 0.8 + 0.6 + 0.75 + 0.65) = 0.7
     $$

**为什么集成更好 / Why Ensemble is Better:**

1. **减少方差 / Variance Reduction:**
   - 单个模型可能过拟合特定训练样本 / Single model may overfit specific training samples
   - 多个模型的平均减少预测方差 / Average of multiple models reduces prediction variance

2. **鲁棒性 / Robustness:**
   - 即使某些模型预测错误，其他模型可以纠正 / Even if some models predict wrong, others can correct
   - 对异常值和噪声更鲁棒 / More robust to outliers and noise

3. **偏差-方差权衡 / Bias-Variance Tradeoff:**
   - Bagging主要减少方差，不增加偏差 / Bagging mainly reduces variance without increasing bias
   - 当基学习器方差高时效果最好 / Works best when base learners have high variance

**结论 / Conclusion:**
Bagging通过集成多个在不同Bootstrap样本上训练的模型，使用投票或平均来做出最终预测，通常比单个模型更准确和稳定。
Bagging makes final prediction by ensemble of multiple models trained on different bootstrap samples using voting or averaging, usually more accurate and stable than single model.

---

### 例题2：AdaBoost权重更新 / AdaBoost Weight Update

**题目 / Question:**  
解释AdaBoost中样本权重如何更新，并说明为什么错误分类的样本权重会增加。
Explain how sample weights are updated in AdaBoost and explain why weights of misclassified samples increase.

**详细解答 / Detailed Solution:**

**AdaBoost权重更新规则 / AdaBoost Weight Update Rule:**
$$
w_i^{(t+1)} = \frac{w_i^{(t)}}{Z_t} \exp(-\alpha_t y^{(i)} h_t(\mathbf{x}^{(i)}))
$$

其中 / where:
- $\alpha_t = \frac{1}{2}\log\frac{1-\epsilon_t}{\epsilon_t}$：学习器权重 / Learner weight
- $Z_t$：归一化常数 / Normalization constant

**权重更新分析 / Weight Update Analysis:**

1. **正确分类的样本 / Correctly Classified Samples:**
   - $y^{(i)} h_t(\mathbf{x}^{(i)}) > 0$（符号相同）/ Same sign
   - $\exp(-\alpha_t \times \text{正数}) = \exp(-\text{正数}) < 1$ / exp(-positive) < 1
   - 权重减小 / Weight decreases

2. **错误分类的样本 / Misclassified Samples:**
   - $y^{(i)} h_t(\mathbf{x}^{(i)}) < 0$（符号相反）/ Opposite sign
   - $\exp(-\alpha_t \times \text{负数}) = \exp(\text{正数}) > 1$ / exp(positive) > 1
   - 权重增加 / Weight increases

**示例 / Example:**
假设 / Suppose:
- $\alpha_t = 1.0$
- 样本1正确分类：$y^{(1)} h_t(\mathbf{x}^{(1)}) = +1$ / Sample 1 correctly classified
- 样本2错误分类：$y^{(2)} h_t(\mathbf{x}^{(2)}) = -1$ / Sample 2 misclassified

权重更新 / Weight update:
- 样本1：$w_1^{(t+1)} \propto w_1^{(t)} \times e^{-1} \approx 0.368 w_1^{(t)}$（减小）/ decreases
- 样本2：$w_2^{(t+1)} \propto w_2^{(t)} \times e^{+1} \approx 2.718 w_2^{(t)}$（增加）/ increases

**结论 / Conclusion:**
AdaBoost通过增加错误分类样本的权重，使后续学习器更关注这些难分类的样本，从而逐步改进模型性能。
AdaBoost improves model performance step by step by increasing weights of misclassified samples, making subsequent learners focus more on these hard-to-classify samples.

---

