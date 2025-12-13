# Lecture 12 知识点总结 / Knowledge Summary

## Table of Contents / 目录

- [1. Model Evaluation / 1. 模型评估](#1-model-evaluation--1-模型评估)
- [2. Cross-Validation / 2. 交叉验证](#2-cross-validation--2-交叉验证)
- [3. Hyperparameter Tuning / 3. 超参数调优](#3-hyperparameter-tuning--3-超参数调优)
- [4. Model Selection / 4. 模型选择](#4-model-selection--4-模型选择)

---

## 1. Model Evaluation / 1. 模型评估

#### English
Model evaluation measures how well a model performs on unseen data. It's crucial for understanding model quality and making informed decisions about model selection and deployment.

#### 中文
模型评估衡量模型在未见数据上的表现。这对于理解模型质量和做出关于模型选择和部署的明智决策至关重要。

**评估指标 / Evaluation Metrics:**

**分类指标 / Classification Metrics:**

1. **准确率（Accuracy）:**
   $$
   \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
   $$

2. **精确率（Precision）:**
   $$
   \text{Precision} = \frac{TP}{TP + FP}
   $$

3. **召回率（Recall）:**
   $$
   \text{Recall} = \frac{TP}{TP + FN}
   $$

4. **F1分数（F1-Score）:**
   $$
   F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
   $$

5. **ROC曲线和AUC / ROC Curve and AUC:**
   - ROC曲线：真阳性率 vs 假阳性率 / ROC curve: True Positive Rate vs False Positive Rate
   - AUC：曲线下面积，范围[0, 1] / AUC: Area under curve, range [0, 1]

**回归指标 / Regression Metrics:**

1. **均方误差（MSE）:**
   $$
   MSE = \frac{1}{M}\sum_{i=1}^M (y^{(i)} - \hat{y}^{(i)})^2
   $$

2. **均方根误差（RMSE）:**
   $$
   RMSE = \sqrt{MSE}
   $$

3. **平均绝对误差（MAE）:**
   $$
   MAE = \frac{1}{M}\sum_{i=1}^M |y^{(i)} - \hat{y}^{(i)}|
   $$

4. **决定系数（R²）:**
   $$
   R^2 = 1 - \frac{\sum_{i=1}^M (y^{(i)} - \hat{y}^{(i)})^2}{\sum_{i=1}^M (y^{(i)} - \bar{y})^2}
   $$

#### 通俗解释
模型评估就像"考试"：训练是"学习"，评估是"考试"。准确率是"答对多少题"，精确率是"说对的时候真的对多少"，召回率是"该找的都找到了多少"。F1是"精确率和召回率的平衡"。

---

## 2. Cross-Validation / 2. 交叉验证

#### English
Cross-validation is a technique to assess model performance by splitting data into multiple folds, training on some folds and testing on others. It provides a more reliable estimate of model performance.

#### 中文
交叉验证是一种通过将数据分成多个折、在一些折上训练并在其他折上测试来评估模型性能的技术。它提供了更可靠的模型性能估计。

**K折交叉验证 / K-Fold Cross-Validation:**

**算法 / Algorithm:**
1. 将数据分成K个折 / Split data into K folds
2. 对$k = 1, \ldots, K$:
   - 使用第$k$折作为验证集，其余作为训练集 / Use fold $k$ as validation set, rest as training set
   - 在训练集上训练模型 / Train model on training set
   - 在验证集上评估模型 / Evaluate model on validation set
   - 记录性能指标 / Record performance metric
3. 计算K次评估的平均性能 / Compute average performance over K evaluations

**留一法交叉验证（LOOCV） / Leave-One-Out Cross-Validation (LOOCV):**
- K = M（样本数）/ K = M (number of samples)
- 每次留一个样本作为验证集 / Leave one sample as validation set each time
- 计算量大但无偏 / Computationally expensive but unbiased

**分层K折交叉验证 / Stratified K-Fold Cross-Validation:**
- 保持每个折中类别比例与原始数据相同 / Maintain same class proportions in each fold as original data
- 适用于不平衡数据集 / Useful for imbalanced datasets

**时间序列交叉验证 / Time Series Cross-Validation:**
- 考虑时间顺序 / Consider temporal order
- 训练集总是在测试集之前 / Training set always before test set

#### 通俗解释
交叉验证就像"多次考试"：把数据分成K份，每次用一份当"考试题"，其他当"练习题"，考K次，最后看平均成绩。这样比只考一次更可靠。

---

## 3. Hyperparameter Tuning / 3. 超参数调优

#### English
Hyperparameter tuning finds the best hyperparameters (parameters not learned during training) for a model. Common methods include grid search, random search, and Bayesian optimization.

#### 中文
超参数调优为模型找到最佳超参数（训练期间不学习的参数）。常见方法包括网格搜索、随机搜索和贝叶斯优化。

**网格搜索（Grid Search） / Grid Search:**
- 定义超参数网格 / Define hyperparameter grid
- 尝试所有组合 / Try all combinations
- 选择性能最好的组合 / Select best performing combination

**随机搜索（Random Search） / Random Search:**
- 从超参数空间中随机采样 / Randomly sample from hyperparameter space
- 通常比网格搜索更高效 / Usually more efficient than grid search
- 可以找到更好的解 / Can find better solutions

**贝叶斯优化（Bayesian Optimization） / Bayesian Optimization:**
- 使用先验知识指导搜索 / Use prior knowledge to guide search
- 更智能地选择下一个要尝试的超参数 / More intelligently select next hyperparameters to try
- 通常需要最少的评估次数 / Usually requires fewest evaluations

**超参数示例 / Hyperparameter Examples:**
- 学习率 / Learning rate
- 正则化系数 / Regularization coefficient
- 树的最大深度 / Maximum tree depth
- 隐藏层大小 / Hidden layer size
- 批量大小 / Batch size

**嵌套交叉验证 / Nested Cross-Validation:**
- 外层：模型选择 / Outer: Model selection
- 内层：超参数调优 / Inner: Hyperparameter tuning
- 避免过拟合到验证集 / Avoid overfitting to validation set

#### 通俗解释
超参数调优就像"调参数"：模型有一些"设置"（超参数）需要手动调整，比如学习率、树的深度等。网格搜索是"试所有组合"，随机搜索是"随机试"，贝叶斯优化是"聪明地试"。

---

## 4. Model Selection / 4. 模型选择

#### English
Model selection involves choosing the best model among different algorithms, architectures, or configurations. It requires careful evaluation and comparison.

#### 中文
模型选择涉及在不同算法、架构或配置中选择最佳模型。它需要仔细评估和比较。

**选择准则 / Selection Criteria:**
- **性能 / Performance**: 在验证集上的表现 / Performance on validation set
- **复杂度 / Complexity**: 模型复杂度（避免过拟合）/ Model complexity (avoid overfitting)
- **可解释性 / Interpretability**: 模型是否容易理解 / Whether model is easy to understand
- **计算成本 / Computational Cost**: 训练和预测时间 / Training and prediction time
- **鲁棒性 / Robustness**: 对数据变化的敏感性 / Sensitivity to data changes

**模型比较 / Model Comparison:**
- 使用相同的评估指标 / Use same evaluation metrics
- 使用相同的交叉验证设置 / Use same cross-validation setup
- 进行统计显著性测试 / Perform statistical significance tests

**偏差-方差权衡 / Bias-Variance Tradeoff:**
- **高偏差 / High Bias**: 模型太简单，欠拟合 / Model too simple, underfitting
- **高方差 / High Variance**: 模型太复杂，过拟合 / Model too complex, overfitting
- **最佳模型 / Best Model**: 偏差和方差的平衡 / Balance between bias and variance

**学习曲线 / Learning Curves:**
- 绘制训练和验证误差 vs 训练样本数 / Plot training and validation error vs number of training samples
- 帮助诊断偏差和方差问题 / Help diagnose bias and variance issues

#### 通俗解释
模型选择就像"选工具"：不同问题需要不同工具。要看：
- **性能**：能不能解决问题
- **复杂度**：会不会太复杂（过拟合）或太简单（欠拟合）
- **实用性**：好不好用、快不快

---

## Additional Detail / 补充要点

### Overfitting and Underfitting / 过拟合和欠拟合

#### English
- **Overfitting**: Model performs well on training data but poorly on test data
  - **Solution**: Regularization, more data, simpler model
- **Underfitting**: Model is too simple to capture patterns
  - **Solution**: More complex model, more features, less regularization

#### 中文
- **过拟合**: 模型在训练数据上表现好但在测试数据上表现差
  - **解决方案**: 正则化、更多数据、更简单的模型
- **欠拟合**: 模型太简单无法捕捉模式
  - **解决方案**: 更复杂的模型、更多特征、更少正则化

---

### Confusion Matrix / 混淆矩阵

#### English
A confusion matrix shows the counts of true positives, false positives, true negatives, and false negatives. It's useful for understanding model performance in detail.

#### 中文
混淆矩阵显示真阳性、假阳性、真阴性和假阴性的计数。它对于详细理解模型性能很有用。

**混淆矩阵 / Confusion Matrix:**
$$
\begin{bmatrix}
TN & FP \\
FN & TP
\end{bmatrix}
$$

---

### Precision-Recall Curve / 精确率-召回率曲线

#### English
The Precision-Recall (PR) curve plots precision vs recall for different thresholds. It's especially useful for imbalanced datasets.

#### 中文
精确率-召回率（PR）曲线绘制不同阈值下的精确率 vs 召回率。它对不平衡数据集特别有用。

---

## Learning Recommendations / 学习建议

### For Beginners / 对于初学者

1. **Understand Evaluation Metrics / 理解评估指标:**
   - Implement from scratch
   - See how they relate
   - Understand when to use each

2. **Practice Cross-Validation / 练习交叉验证:**
   - Implement K-fold CV
   - Compare with train-test split
   - Understand bias-variance tradeoff

3. **Tune Hyperparameters / 调优超参数:**
   - Use grid search
   - Try random search
   - Compare results

---

## Reference Resources / 参考资源

1. **Textbooks / 教科书:**
   - The Elements of Statistical Learning (Hastie et al.)
   - Pattern Recognition and Machine Learning (Bishop)

2. **Online Resources / 在线资源:**
   - Scikit-learn Model Selection Documentation
   - Cross-Validation Best Practices

---

## 练习题 / Practice Problems

### 问题1 / Problem 1:
解释为什么需要将数据分为训练集、验证集和测试集，并说明各自的用途。
Explain why data needs to be split into training, validation, and test sets, and explain their respective purposes.

### 问题2 / Problem 2:
比较K折交叉验证和留一法交叉验证的优缺点。
Compare the advantages and disadvantages of K-fold cross-validation and leave-one-out cross-validation.

### 问题3 / Problem 3:
解释如何通过学习曲线诊断模型的偏差和方差问题。
Explain how to diagnose bias and variance problems through learning curves.

---

## 例题与解答 / Worked Examples

### 例题1：K折交叉验证计算 / K-Fold Cross-Validation Calculation

**题目 / Question:**  
给定10个样本，使用5折交叉验证评估模型。说明如何划分数据并计算最终性能。
Given 10 samples, use 5-fold cross-validation to evaluate model. Explain how to split data and calculate final performance.

**详细解答 / Detailed Solution:**

**步骤1：划分数据 / Step 1: Split Data**
将10个样本分成5折，每折2个样本 / Split 10 samples into 5 folds, 2 samples per fold:
- 折1：样本1, 2 / Fold 1: samples 1, 2
- 折2：样本3, 4 / Fold 2: samples 3, 4
- 折3：样本5, 6 / Fold 3: samples 5, 6
- 折4：样本7, 8 / Fold 4: samples 7, 8
- 折5：样本9, 10 / Fold 5: samples 9, 10

**步骤2：交叉验证过程 / Step 2: Cross-Validation Process**

迭代1 / Iteration 1:
- 训练集：折2-5（样本3-10）/ Training set: folds 2-5 (samples 3-10)
- 验证集：折1（样本1, 2）/ Validation set: fold 1 (samples 1, 2)
- 性能：$\text{accuracy}_1 = 0.9$ / Performance: $\text{accuracy}_1 = 0.9$

迭代2 / Iteration 2:
- 训练集：折1, 3-5（样本1,2,5-10）/ Training set: folds 1, 3-5 (samples 1,2,5-10)
- 验证集：折2（样本3, 4）/ Validation set: fold 2 (samples 3, 4)
- 性能：$\text{accuracy}_2 = 0.85$ / Performance: $\text{accuracy}_2 = 0.85$

迭代3 / Iteration 3:
- 训练集：折1-2, 4-5 / Training set: folds 1-2, 4-5
- 验证集：折3 / Validation set: fold 3
- 性能：$\text{accuracy}_3 = 0.95$ / Performance: $\text{accuracy}_3 = 0.95$

迭代4 / Iteration 4:
- 训练集：折1-3, 5 / Training set: folds 1-3, 5
- 验证集：折4 / Validation set: fold 4
- 性能：$\text{accuracy}_4 = 0.9$ / Performance: $\text{accuracy}_4 = 0.9$

迭代5 / Iteration 5:
- 训练集：折1-4 / Training set: folds 1-4
- 验证集：折5 / Validation set: fold 5
- 性能：$\text{accuracy}_5 = 0.88$ / Performance: $\text{accuracy}_5 = 0.88$

**步骤3：计算平均性能 / Step 3: Calculate Average Performance**
$$
\text{平均准确率} = \frac{1}{5}(0.9 + 0.85 + 0.95 + 0.9 + 0.88) = \frac{4.48}{5} = 0.896
$$

**结论 / Conclusion:**
5折交叉验证的平均准确率为 $89.6\%$，这比单次训练-测试划分更可靠地估计了模型性能。
5-fold cross-validation average accuracy is $89.6\%$, which more reliably estimates model performance than single train-test split.

---

### 例题2：学习曲线诊断 / Learning Curve Diagnosis

**题目 / Question:**  
解释如何通过学习曲线判断模型是否存在高偏差或高方差问题。
Explain how to diagnose high bias or high variance problems through learning curves.

**详细解答 / Detailed Solution:**

**学习曲线 / Learning Curves:**
绘制训练误差和验证误差随训练样本数的变化 / Plot training and validation errors vs number of training samples

**高偏差（欠拟合）的特征 / Characteristics of High Bias (Underfitting):**

1. **训练误差高 / High Training Error:**
   - 模型太简单，无法拟合训练数据 / Model too simple, cannot fit training data
   - 训练误差接近验证误差 / Training error close to validation error

2. **验证误差高 / High Validation Error:**
   - 模型性能差，即使增加数据也不会改善 / Poor model performance, won't improve even with more data

3. **曲线特征 / Curve Characteristics:**
   - 两条曲线都高且接近 / Both curves high and close
   - 增加数据后误差下降很小 / Error decreases little with more data

**解决方案 / Solutions:**
- 增加模型复杂度 / Increase model complexity
- 添加更多特征 / Add more features
- 减少正则化 / Reduce regularization

**高方差（过拟合）的特征 / Characteristics of High Variance (Overfitting):**

1. **训练误差低 / Low Training Error:**
   - 模型很好地拟合了训练数据 / Model fits training data well
   - 训练误差远低于验证误差 / Training error much lower than validation error

2. **验证误差高 / High Validation Error:**
   - 模型泛化能力差 / Poor generalization ability
   - 训练误差和验证误差之间有较大差距 / Large gap between training and validation error

3. **曲线特征 / Curve Characteristics:**
   - 训练误差低，验证误差高 / Low training error, high validation error
   - 增加数据后验证误差可能下降 / Validation error may decrease with more data

**解决方案 / Solutions:**
- 增加正则化 / Increase regularization
- 减少模型复杂度 / Reduce model complexity
- 收集更多训练数据 / Collect more training data
- 使用集成方法 / Use ensemble methods

**结论 / Conclusion:**
学习曲线是诊断模型问题的有力工具。通过观察训练误差和验证误差的关系，可以判断是偏差问题还是方差问题，并采取相应的解决措施。
Learning curves are powerful tools for diagnosing model problems. By observing the relationship between training and validation errors, we can determine whether it's a bias or variance problem and take appropriate measures.

---

