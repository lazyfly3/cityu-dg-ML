# Lecture 4 知识点总结 / Knowledge Summary

## Table of Contents / 目录

- [1. Naive Bayes / 1. 朴素贝叶斯](#1-naive-bayes--1-朴素贝叶斯)
- [2. Bayesian Inference / 2. 贝叶斯推断](#2-bayesian-inference--2-贝叶斯推断)
- [3. Generative vs Discriminative / 3. 生成式 vs 判别式](#3-generative-vs-discriminative--3-生成式-vs-判别式)
- [4. Gaussian Naive Bayes / 4. 高斯朴素贝叶斯](#4-gaussian-naive-bayes--4-高斯朴素贝叶斯)

---

## 1. Naive Bayes / 1. 朴素贝叶斯

#### English
Naive Bayes is a probabilistic classifier based on Bayes' theorem with the "naive" assumption of feature independence. It's simple, fast, and works well for many classification problems.

#### 中文
朴素贝叶斯是一种基于贝叶斯定理的概率分类器，具有特征独立的"朴素"假设。它简单、快速，在许多分类问题上表现良好。

**数学定义 / Mathematical Definition:**

**贝叶斯定理 / Bayes' Theorem:**
$$
P(y|\mathbf{x}) = \frac{P(\mathbf{x}|y) P(y)}{P(\mathbf{x})}
$$

**朴素假设 / Naive Assumption:**
假设特征之间相互独立 / Assume features are independent:
$$
P(\mathbf{x}|y) = P(x_1, x_2, \ldots, x_n|y) = \prod_{j=1}^n P(x_j|y)
$$

**朴素贝叶斯分类器 / Naive Bayes Classifier:**
$$
P(y|\mathbf{x}) = \frac{P(y) \prod_{j=1}^n P(x_j|y)}{P(\mathbf{x})} \propto P(y) \prod_{j=1}^n P(x_j|y)
$$

**预测规则 / Prediction Rule:**
$$
\hat{y} = \arg\max_y P(y) \prod_{j=1}^n P(x_j|y)
$$

**符号说明 / Symbol Explanation:**
- $P(y)$：类别y的先验概率 / Prior probability of class y
- $P(x_j|y)$：在类别y下，特征$x_j$的条件概率 / Conditional probability of feature $x_j$ given class y
- $\prod_{j=1}^n$：对所有特征求乘积 / Product over all features

**计算步骤 / Calculation Steps:**
1. 估计先验概率：$P(y) = \frac{\text{类别y的样本数}}{\text{总样本数}}$ / Estimate prior: $P(y) = \frac{\text{count of class y}}{\text{total samples}}$
2. 估计条件概率：$P(x_j|y) = \frac{\text{类别y中特征$x_j$出现的次数}}{\text{类别y的样本数}}$ / Estimate conditional: $P(x_j|y) = \frac{\text{count of $x_j$ in class y}}{\text{count of class y}}$
3. 对新样本，计算每个类别的后验概率 / For new sample, calculate posterior for each class
4. 选择概率最大的类别 / Choose class with highest probability

#### 通俗解释
朴素贝叶斯就像"数数"：统计每个类别下每个特征出现的频率，然后用这些频率来预测新样本的类别。"朴素"假设是说特征之间互不影响（虽然现实中可能不是这样，但这样假设让计算变简单）。

---

## 2. Bayesian Inference / 2. 贝叶斯推断

#### English
Bayesian inference updates prior beliefs about parameters using observed data to obtain posterior distributions. It provides a principled way to incorporate uncertainty.

#### 中文
贝叶斯推断使用观测数据更新参数的先验信念，得到后验分布。它提供了一种原则性的方法来纳入不确定性。

**贝叶斯更新 / Bayesian Update:**

**先验 / Prior:**
$$
P(\theta)
$$

**似然 / Likelihood:**
$$
P(\mathcal{D}|\theta)
$$

**后验 / Posterior:**
$$
P(\theta|\mathcal{D}) = \frac{P(\mathcal{D}|\theta) P(\theta)}{P(\mathcal{D})} \propto P(\mathcal{D}|\theta) P(\theta)
$$

其中 / where:
- $\theta$：模型参数 / Model parameters
- $\mathcal{D}$：观测数据 / Observed data
- $P(\mathcal{D}) = \int P(\mathcal{D}|\theta) P(\theta) d\theta$：证据（归一化常数）/ Evidence (normalization constant)

**最大后验估计（MAP）/ Maximum A Posteriori (MAP):**
$$
\theta_{\text{MAP}} = \arg\max_\theta P(\theta|\mathcal{D}) = \arg\max_\theta P(\mathcal{D}|\theta) P(\theta)
$$

#### 通俗解释
贝叶斯推断就像"更新信念"：先有一个初始猜测（先验），看到数据后更新这个猜测（后验）。MAP估计是找"最可能"的参数值，既考虑数据（似然），也考虑先验知识。

---

## 3. Generative vs Discriminative / 3. 生成式 vs 判别式

#### English
- **Generative models**: Model the joint distribution $P(\mathbf{x}, y)$ and use it to compute $P(y|\mathbf{x})$
- **Discriminative models**: Directly model the conditional distribution $P(y|\mathbf{x})$

#### 中文
- **生成式模型**：建模联合分布$P(\mathbf{x}, y)$并用它计算$P(y|\mathbf{x})$
- **判别式模型**：直接建模条件分布$P(y|\mathbf{x})$

**生成式模型示例 / Generative Model Example:**
- 朴素贝叶斯 / Naive Bayes
- 高斯判别分析 / Gaussian Discriminant Analysis
- 隐马尔可夫模型 / Hidden Markov Models

**判别式模型示例 / Discriminative Model Example:**
- 逻辑回归 / Logistic Regression
- 支持向量机 / Support Vector Machines
- 神经网络 / Neural Networks

**比较 / Comparison:**

| 特性 / Feature | 生成式 / Generative | 判别式 / Discriminative |
|--------------|-------------------|----------------------|
| 建模 / Models | $P(\mathbf{x}, y)$ | $P(y|\mathbf{x})$ |
| 优点 / Pros | 可以生成样本 / Can generate samples | 通常分类性能更好 / Usually better classification |
| 缺点 / Cons | 需要更多假设 / Needs more assumptions | 不能生成样本 / Cannot generate samples |

#### 通俗解释
生成式模型像"学会画图"：学会每个类别长什么样，然后判断新样本像哪个类别。判别式模型像"学会区分"：直接学会如何区分不同类别，不管每个类别具体长什么样。

---

## 4. Gaussian Naive Bayes / 4. 高斯朴素贝叶斯

#### English
Gaussian Naive Bayes assumes that continuous features follow a Gaussian (normal) distribution for each class. It's useful for continuous-valued features.

#### 中文
高斯朴素贝叶斯假设连续特征在每个类别下服从高斯（正态）分布。它适用于连续值特征。

**数学定义 / Mathematical Definition:**

**高斯分布 / Gaussian Distribution:**
$$
P(x_j|y) = \mathcal{N}(x_j; \mu_{jy}, \sigma_{jy}^2) = \frac{1}{\sqrt{2\pi\sigma_{jy}^2}} \exp\left(-\frac{(x_j - \mu_{jy})^2}{2\sigma_{jy}^2}\right)
$$

**参数估计 / Parameter Estimation:**

**均值 / Mean:**
$$
\mu_{jy} = \frac{1}{M_y} \sum_{i:y^{(i)}=y} x_j^{(i)}
$$

**方差 / Variance:**
$$
\sigma_{jy}^2 = \frac{1}{M_y} \sum_{i:y^{(i)}=y} (x_j^{(i)} - \mu_{jy})^2
$$

其中 / where:
- $M_y$：类别y的样本数 / Number of samples in class y
- $\mu_{jy}$：类别y下特征$j$的均值 / Mean of feature $j$ in class y
- $\sigma_{jy}^2$：类别y下特征$j$的方差 / Variance of feature $j$ in class y

**对数概率（数值稳定）/ Log Probability (Numerically Stable):**
$$
\log P(y|\mathbf{x}) = \log P(y) + \sum_{j=1}^n \log P(x_j|y) - \log P(\mathbf{x})
$$

$$
= \log P(y) + \sum_{j=1}^n \left[-\frac{1}{2}\log(2\pi\sigma_{jy}^2) - \frac{(x_j - \mu_{jy})^2}{2\sigma_{jy}^2}\right] + \text{const}
$$

**计算步骤 / Calculation Steps:**
1. 对每个类别y，估计先验$P(y)$ / For each class y, estimate prior $P(y)$
2. 对每个类别y和每个特征j，估计$\mu_{jy}$和$\sigma_{jy}^2$ / For each class y and feature j, estimate $\mu_{jy}$ and $\sigma_{jy}^2$
3. 对新样本，计算每个类别的对数后验概率 / For new sample, calculate log posterior for each class
4. 选择概率最大的类别 / Choose class with highest probability

#### 通俗解释
高斯朴素贝叶斯假设每个特征在每个类别下都像一个"钟形曲线"（高斯分布）。我们估计每个类别下每个特征的"中心"（均值）和"宽度"（方差），然后用这些来预测新样本的类别。

---

## Additional Detail / 补充要点

### Laplace Smoothing / 拉普拉斯平滑

#### English
Add a small constant to avoid zero probabilities:
$$
P(x_j|y) = \frac{\text{count}(x_j, y) + \alpha}{\text{count}(y) + \alpha |V|}
$$

where $\alpha$ is the smoothing parameter and $|V|$ is the vocabulary size.

#### 中文
添加小常数避免零概率：
$$
P(x_j|y) = \frac{\text{count}(x_j, y) + \alpha}{\text{count}(y) + \alpha |V|}
$$

其中$\alpha$是平滑参数，$|V|$是词汇表大小。

---

### Multinomial Naive Bayes / 多项式朴素贝叶斯

#### English
For discrete features (e.g., word counts in text):
$$
P(\mathbf{x}|y) = \frac{(\sum_j x_j)!}{\prod_j x_j!} \prod_j P(x_j|y)^{x_j}
$$

#### 中文
对于离散特征（如文本中的词频）：
$$
P(\mathbf{x}|y) = \frac{(\sum_j x_j)!}{\prod_j x_j!} \prod_j P(x_j|y)^{x_j}
$$

---

### Advantages and Disadvantages / 优缺点

#### English
**Advantages:**
- Simple and fast
- Works well with small datasets
- Handles multiple classes naturally
- Less prone to overfitting

**Disadvantages:**
- Strong independence assumption
- May not capture feature interactions
- Performance can be limited

#### 中文
**优点:**
- 简单快速
- 在小数据集上表现好
- 自然处理多类问题
- 不易过拟合

**缺点:**
- 强独立性假设
- 可能无法捕捉特征交互
- 性能可能受限

---

## Learning Recommendations / 学习建议

### For Beginners / 对于初学者

1. **Understand Bayes' Theorem / 理解贝叶斯定理:**
   - Work through examples
   - See how prior and likelihood combine
   - Understand posterior interpretation

2. **Implement Naive Bayes / 实现朴素贝叶斯:**
   - For discrete features
   - For continuous features (Gaussian)
   - With Laplace smoothing

3. **Compare with Logistic Regression / 与逻辑回归比较:**
   - Understand differences
   - See when each works better
   - Experiment with both

---

## Reference Resources / 参考资源

1. **Textbooks / 教科书:**
   - Pattern Recognition and Machine Learning (Bishop)
   - Machine Learning: A Probabilistic Perspective (Murphy)

2. **Online Resources / 在线资源:**
   - Stanford CS229 Lecture Notes
   - Scikit-learn Naive Bayes Documentation

---

## 练习题 / Practice Problems

### 问题1 / Problem 1:
解释朴素贝叶斯中的"朴素"假设，并说明为什么这个假设在实际中可能不成立。
Explain the "naive" assumption in Naive Bayes and explain why this assumption may not hold in practice.

### 问题2 / Problem 2:
推导高斯朴素贝叶斯的参数估计公式（均值和方差）。
Derive the parameter estimation formulas (mean and variance) for Gaussian Naive Bayes.

### 问题3 / Problem 3:
比较生成式模型和判别式模型的优缺点，并各举一个例子。
Compare the advantages and disadvantages of generative and discriminative models, and give one example of each.

---

## 例题与解答 / Worked Examples

### 例题1：朴素贝叶斯分类 / Naive Bayes Classification

**题目 / Question:**  
给定训练数据，使用朴素贝叶斯进行分类。训练数据如下：
Given training data, use Naive Bayes for classification. Training data:

| 样本 | 特征1 | 特征2 | 类别 |
|------|-------|-------|------|
| 1 | 1 | 0 | A |
| 2 | 1 | 1 | A |
| 3 | 0 | 1 | B |
| 4 | 0 | 0 | B |

预测新样本(特征1=1, 特征2=1)的类别。
Predict the class of new sample (feature1=1, feature2=1).

**详细解答 / Detailed Solution:**

**步骤1：估计先验概率 / Step 1: Estimate Prior Probabilities**
- P(A) = 2/4 = 0.5
- P(B) = 2/4 = 0.5

**步骤2：估计条件概率 / Step 2: Estimate Conditional Probabilities**

对于类别A / For class A:
- P(特征1=1|A) = 2/2 = 1.0
- P(特征1=0|A) = 0/2 = 0.0
- P(特征2=1|A) = 1/2 = 0.5
- P(特征2=0|A) = 1/2 = 0.5

对于类别B / For class B:
- P(特征1=1|B) = 0/2 = 0.0
- P(特征1=0|B) = 2/2 = 1.0
- P(特征2=1|B) = 1/2 = 0.5
- P(特征2=0|B) = 1/2 = 0.5

**步骤3：计算后验概率 / Step 3: Calculate Posterior Probabilities**

对于新样本(1,1) / For new sample (1,1):

P(A|特征1=1, 特征2=1) ∝ P(A) × P(特征1=1|A) × P(特征2=1|A)
= 0.5 × 1.0 × 0.5 = 0.25

P(B|特征1=1, 特征2=1) ∝ P(B) × P(特征1=1|B) × P(特征2=1|B)
= 0.5 × 0.0 × 0.5 = 0.0

**步骤4：归一化并预测 / Step 4: Normalize and Predict**
- P(A|新样本) = 0.25 / (0.25 + 0.0) = 1.0
- P(B|新样本) = 0.0 / (0.25 + 0.0) = 0.0

**结论 / Conclusion:**
预测类别为A。
Predicted class is A.

---

### 例题2：贝叶斯更新 / Bayesian Update

**题目 / Question:**  
假设先验概率P(疾病)=0.01，检测准确率P(阳性|疾病)=0.99，P(阴性|健康)=0.95。如果检测结果为阳性，求患病的后验概率。
Suppose prior probability P(disease)=0.01, test accuracy P(positive|disease)=0.99, P(negative|healthy)=0.95. If test result is positive, find posterior probability of having disease.

**详细解答 / Detailed Solution:**

**步骤1：使用贝叶斯定理 / Step 1: Use Bayes' Theorem**
$$
P(\text{疾病}|\text{阳性}) = \frac{P(\text{阳性}|\text{疾病}) P(\text{疾病})}{P(\text{阳性})}
$$

**步骤2：计算P(阳性) / Step 2: Calculate P(positive)**
使用全概率公式 / Using law of total probability:
$$
P(\text{阳性}) = P(\text{阳性}|\text{疾病})P(\text{疾病}) + P(\text{阳性}|\text{健康})P(\text{健康})
$$

$$
P(\text{阳性}) = 0.99 \times 0.01 + (1-0.95) \times 0.99 = 0.0099 + 0.0495 = 0.0594
$$

**步骤3：计算后验概率 / Step 3: Calculate Posterior Probability**
$$
P(\text{疾病}|\text{阳性}) = \frac{0.99 \times 0.01}{0.0594} = \frac{0.0099}{0.0594} \approx 0.167
$$

**结论 / Conclusion:**
即使检测为阳性，患病的概率也只有约16.7%，这是因为疾病的先验概率很低。
Even with positive test, probability of disease is only about 16.7%, because prior probability of disease is very low.

---

