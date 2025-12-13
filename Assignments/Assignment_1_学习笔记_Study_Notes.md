# Assignment 1 学习笔记 / Study Notes

## 目录 / Table of Contents
1. [概率论基础 / Probability Fundamentals](#概率论基础)
2. [朴素贝叶斯 / Naive Bayes](#朴素贝叶斯)
3. [线性判别分析 / Linear Discriminant Analysis](#线性判别分析)
4. [KL散度 / KL Divergence](#kl散度)

---

## 概率论基础 / Probability Fundamentals

### 1.1 期望值 / Expected Value

**中文解释：**
期望值是随机变量所有可能取值的加权平均，权重是每个值出现的概率。

**English Explanation:**
The expected value is the weighted average of all possible values of a random variable, where the weights are the probabilities of each value.

**数学定义 / Mathematical Definition:**  
$$
E[X] = \sum_{x\in\mathcal{X}} x \cdot P(X = x)
$$

**符号说明 / Symbol Explanation:**
- $E[X]$：随机变量X的期望值 / Expected value of random variable X
- $\sum_{x\in\mathcal{X}}$：对所有可能的x值求和 / Sum over all possible values of x
- $x$：随机变量的某个取值 / A specific value of the random variable
- $P(X = x)$：X取值为x的概率 / Probability that X equals x
- $\mathcal{X}$：X所有可能取值的集合 / Set of all possible values of X

**计算步骤 / Calculation Steps:**
1. 列出X所有可能的取值 / List all possible values of X
2. 对每个取值x，计算 x × P(X=x) / For each value x, calculate x × P(X=x)
3. 将所有结果相加 / Sum all the results

**计算示例 / Calculation Example:**
假设一个不公平的骰子，各面概率为：P(1)=0.1, P(2)=0.1, P(3)=0.2, P(4)=0.2, P(5)=0.4, P(6)=0
Suppose an unfair die with probabilities: P(1)=0.1, P(2)=0.1, P(3)=0.2, P(4)=0.2, P(5)=0.4, P(6)=0

计算期望值 / Calculate expected value:
- E[X] = 1×0.1 + 2×0.1 + 3×0.2 + 4×0.2 + 5×0.4 + 6×0
- E[X] = 0.1 + 0.2 + 0.6 + 0.8 + 2.0 + 0
- E[X] = 3.7

**示例 / Example:**
- 公平骰子：E[X] = 1×1/6 + 2×1/6 + 3×1/6 + 4×1/6 + 5×1/6 + 6×1/6 = 21/6 = 3.5
- Fair die: E[X] = 1×1/6 + 2×1/6 + 3×1/6 + 4×1/6 + 5×1/6 + 6×1/6 = 21/6 = 3.5

### 1.2 指示函数 / Indicator Function

**中文解释：**
指示函数 I[X = a] 是一个特殊的函数，当事件发生时值为1，否则为0。

**English Explanation:**
The indicator function I[X = a] is a special function that equals 1 when the event occurs, and 0 otherwise.

**定义 / Definition:**  
$$
I[X=a] = 
\begin{cases}
1, & X=a \\
0, & \text{otherwise}
\end{cases}
$$

**重要性质 / Important Property:**  
$$
E[I[X=a]] = P(X=a)
$$

**计算示例 / Calculation Example:**
假设X可以取{3, 8, 9}，概率分别为P(3)=0.3, P(8)=0.5, P(9)=0.2
Suppose X can take {3, 8, 9} with probabilities P(3)=0.3, P(8)=0.5, P(9)=0.2

计算E[I[X=8]] / Calculate E[I[X=8]]:
- 当X=8时，I[X=8]=1，概率为0.5 / When X=8, I[X=8]=1 with probability 0.5
- 当X≠8时（X=3或9），I[X=8]=0，概率为0.3+0.2=0.5 / When X≠8 (X=3 or 9), I[X=8]=0 with probability 0.3+0.2=0.5
- E[I[X=8]] = 1×0.5 + 0×0.5 = 0.5 = P(X=8) ✓

**证明 / Proof:**  
$$
E[I[X=a]] = 1\cdot P(X=a) + 0\cdot P(X\neq a) = P(X=a)
$$

### 1.3 熵 / Entropy

**中文解释：**
熵衡量随机变量的不确定性或信息量。熵越大，不确定性越大。

**English Explanation:**
Entropy measures the uncertainty or information content of a random variable. Higher entropy means greater uncertainty.

**定义 / Definition:**  
$$
H(X) = -\sum_{x\in\mathcal{X}} P(X=x)\log_2 P(X=x) = -E[\log_2 P(X)]
$$

**符号说明 / Symbol Explanation:**
- $H(X)$：随机变量X的熵 / Entropy of random variable X
- $\sum_{x\in\mathcal{X}}$：对所有可能的x值求和 / Sum over all possible values
- $P(X=x)$：X取值为x的概率 / Probability that X equals x
- $\log_2$：以2为底的对数（单位是比特/bit）/ Base-2 logarithm (unit is bits)
- 负号：确保熵为非负值 / Negative sign ensures non-negative entropy

**计算步骤 / Calculation Steps:**
1. 对每个可能的x值，计算 $P(X=x) \times \log_2 P(X=x)$ / For each x, calculate $P(X=x) \times \log_2 P(X=x)$
2. 将所有结果相加 / Sum all results
3. 取负号 / Take negative sign

**计算示例 / Calculation Example:**
假设一个不公平硬币，P(正面)=0.7, P(反面)=0.3
Suppose an unfair coin with P(heads)=0.7, P(tails)=0.3

计算熵 / Calculate entropy:
- H(X) = -[P(正面)×log₂(0.7) + P(反面)×log₂(0.3)]
- H(X) = -[0.7×log₂(0.7) + 0.3×log₂(0.3)]
- log₂(0.7) ≈ -0.515, log₂(0.3) ≈ -1.737
- H(X) = -[0.7×(-0.515) + 0.3×(-1.737)]
- H(X) = -[-0.3605 - 0.5211]
- H(X) = -[-0.8816] = 0.8816 bits

公平硬币（P=0.5）的熵为1 bit，所以不公平硬币的熵更小（不确定性更小）
Fair coin (P=0.5) has entropy 1 bit, so unfair coin has lower entropy (less uncertainty)

**性质 / Properties:**
- 熵总是非负的 / Entropy is always non-negative
- 当所有结果等概率时，熵最大 / Entropy is maximized when all outcomes are equally likely
- 当只有一个确定结果时，熵为0 / Entropy is 0 when there's only one certain outcome

**示例 / Example:**
公平硬币：H(X) = -0.5×log₂(0.5) - 0.5×log₂(0.5) = 1 bit
Fair coin: H(X) = -0.5×log₂(0.5) - 0.5×log₂(0.5) = 1 bit

### 1.4 联合熵与条件熵 / Joint and Conditional Entropy

**联合熵 / Joint Entropy:**  
$$
H(X,Y) = -\sum_{x,y} P(X=x, Y=y)\,\log_2 P(X=x, Y=y)
$$

**计算示例 / Calculation Example:**
假设X和Y的联合分布如下：
Suppose joint distribution of X and Y:

|     | Y=0 | Y=1 |
|-----|-----|-----|
| X=0 | 0.3 | 0.2 |
| X=1 | 0.1 | 0.4 |

计算联合熵 / Calculate joint entropy:
- H(X,Y) = -[0.3×log₂(0.3) + 0.2×log₂(0.2) + 0.1×log₂(0.1) + 0.4×log₂(0.4)]
- H(X,Y) = -[0.3×(-1.737) + 0.2×(-2.322) + 0.1×(-3.322) + 0.4×(-1.322)]
- H(X,Y) = -[-0.521 - 0.464 - 0.332 - 0.529] = 1.846 bits

**条件熵 / Conditional Entropy:**  
$$
H(Y|X) = -\sum_{x,y} P(X=x, Y=y)\,\log_2 P(Y=y\mid X=x)
$$

**计算示例 / Calculation Example:**
使用上面的联合分布，先计算条件概率：
Using above joint distribution, first calculate conditional probabilities:
- P(Y=0|X=0) = 0.3/(0.3+0.2) = 0.6, P(Y=1|X=0) = 0.2/0.5 = 0.4
- P(Y=0|X=1) = 0.1/(0.1+0.4) = 0.2, P(Y=1|X=1) = 0.4/0.5 = 0.8

计算条件熵 / Calculate conditional entropy:
- H(Y|X) = -[0.3×log₂(0.6) + 0.2×log₂(0.4) + 0.1×log₂(0.2) + 0.4×log₂(0.8)]
- H(Y|X) = -[0.3×(-0.737) + 0.2×(-1.322) + 0.1×(-2.322) + 0.4×(-0.322)]
- H(Y|X) = -[-0.221 - 0.264 - 0.232 - 0.129] = 0.846 bits

**链式法则 / Chain Rule:**  
$$
H(X,Y) = H(Y) + H(X|Y) = H(X) + H(Y|X)
$$

**证明思路 / Proof Sketch:**
使用条件概率的定义和期望的线性性质。
Using the definition of conditional probability and linearity of expectation.

### 1.5 互信息 / Mutual Information

**中文解释：**
互信息衡量两个随机变量之间的相互依赖程度。

**English Explanation:**
Mutual information measures the mutual dependence between two random variables.

**定义 / Definition:**  
$$
I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)
$$

**计算示例 / Calculation Example:**
使用上面的联合分布，计算互信息：
Using above joint distribution, calculate mutual information:

首先计算边际分布 / First calculate marginal distributions:
- P(X=0) = 0.3+0.2 = 0.5, P(X=1) = 0.1+0.4 = 0.5
- P(Y=0) = 0.3+0.1 = 0.4, P(Y=1) = 0.2+0.4 = 0.6

计算H(X)和H(Y) / Calculate H(X) and H(Y):
- H(X) = -[0.5×log₂(0.5) + 0.5×log₂(0.5)] = -[-0.5 - 0.5] = 1 bit
- H(Y) = -[0.4×log₂(0.4) + 0.6×log₂(0.6)] = -[-0.529 - 0.442] = 0.971 bits

从前面已知 / From above:
- H(X,Y) = 1.846 bits
- H(Y|X) = 0.846 bits

计算互信息 / Calculate mutual information:
- I(X;Y) = H(Y) - H(Y|X) = 0.971 - 0.846 = 0.125 bits
- 或 / or: I(X;Y) = H(X) + H(Y) - H(X,Y) = 1 + 0.971 - 1.846 = 0.125 bits ✓

**重要性质 / Important Property:**
如果 X 和 Y 独立，则 I(X; Y) = 0
If X and Y are independent, then I(X; Y) = 0

**证明 / Proof:**
如果 X 和 Y 独立，则 P(X=x, Y=y) = P(X=x)P(Y=y)
If X and Y are independent, then P(X=x, Y=y) = P(X=x)P(Y=y)

因此 / Therefore:
```
I(X; Y) = Σ(x,y) P(X=x, Y=y) log₂ [P(X=x, Y=y) / (P(X=x)P(Y=y))]
        = Σ(x,y) P(X=x, Y=y) log₂ 1
        = 0
```

---

## 朴素贝叶斯 / Naive Bayes

### 2.1 基本思想 / Basic Idea

**中文解释：**
朴素贝叶斯假设特征之间相互独立，使用贝叶斯定理进行分类。

**English Explanation:**
Naive Bayes assumes features are independent and uses Bayes' theorem for classification.

**贝叶斯定理 / Bayes' Theorem:**  
$$
P(Y\mid X) = \frac{P(X\mid Y)\,P(Y)}{P(X)}
$$

### 2.2 最大似然估计 / Maximum Likelihood Estimation

**中文解释：**
最大似然估计是选择使观测数据出现概率最大的参数值。

**English Explanation:**
Maximum likelihood estimation chooses parameter values that maximize the probability of observing the data.

**🔰 零基础理解：什么是"似然"？/ Zero-Basics: What is "Likelihood"?**

**通俗解释 / Intuitive Explanation:**
想象你有一个"魔法盒子"，里面有一些参数（比如概率值）。你往盒子里放数据，盒子会告诉你"这些数据出现的可能性有多大"。

Imagine you have a "magic box" with some parameters (like probability values). You put data into the box, and it tells you "how likely these data are to appear."

- **似然（Likelihood）** = 在给定参数下，观测到这些数据的"可能性"
- **Likelihood** = The "possibility" of observing these data given the parameters
- **最大似然估计** = 找到让这个"可能性"最大的参数值
- **Maximum Likelihood Estimation** = Find parameter values that make this "possibility" the largest

**生活类比 / Life Analogy:**
- 你看到3个苹果都是红色的
- You see 3 apples, all are red
- 如果盒子参数说"红苹果概率=0.9"，那么看到3个红苹果的可能性 = 0.9 × 0.9 × 0.9 = 0.729
- If box parameters say "red apple probability = 0.9", then possibility of seeing 3 red apples = 0.9 × 0.9 × 0.9 = 0.729
- 如果盒子参数说"红苹果概率=0.5"，那么可能性 = 0.5 × 0.5 × 0.5 = 0.125
- If box parameters say "red apple probability = 0.5", then possibility = 0.5 × 0.5 × 0.5 = 0.125
- 显然0.9的参数更"合理"（因为实际看到3个都是红的）
- Obviously 0.9 parameter is more "reasonable" (because we actually see 3 red ones)

**似然函数 / Likelihood Function:**  
$$
L(\theta) = \prod_{i=1}^M P\big(x^{(i)}, y^{(i)} \mid \theta\big)
$$

**🔰 零基础：符号详解 / Zero-Basics: Symbol Details**

**每个符号的通俗解释 / Intuitive Explanation of Each Symbol:**

1. **$L(\theta)$ - 似然函数 / Likelihood Function**
   - **通俗理解**：一个"评分函数"，告诉你参数θ有多"好"
   - **Intuitive**: A "scoring function" that tells how "good" parameter θ is
   - **例子**：如果L(θ₁) = 0.8, L(θ₂) = 0.3，说明θ₁比θ₂更好
   - **Example**: If L(θ₁) = 0.8, L(θ₂) = 0.3, then θ₁ is better than θ₂

2. **$\prod_{i=1}^M$ - 连乘符号 / Product Symbol**
   - **通俗理解**：把所有东西"乘起来"
   - **Intuitive**: "Multiply" everything together
   - **例子**：$\prod_{i=1}^{3} a_i = a_1 × a_2 × a_3$
   - **Example**: $\prod_{i=1}^{3} a_i = a_1 × a_2 × a_3$
   - **为什么用乘？**：因为每个样本是"独立"出现的（一个出现不影响另一个）
   - **Why multiply?**: Because each sample appears "independently" (one doesn't affect another)

3. **$M$ - 样本数量 / Number of Samples**
   - **通俗理解**：你有多少个训练数据
   - **Intuitive**: How many training data you have
   - **例子**：M=3 表示有3个样本
   - **Example**: M=3 means 3 samples

4. **$x^{(i)}$ - 第i个样本的特征 / Features of i-th Sample**
   - **通俗理解**：第i个样本的"描述信息"
   - **Intuitive**: "Description information" of i-th sample
   - **例子**：x^(1) = (1, 2) 表示第1个样本有2个特征，值分别是1和2
   - **Example**: x^(1) = (1, 2) means 1st sample has 2 features with values 1 and 2

5. **$y^{(i)}$ - 第i个样本的标签 / Label of i-th Sample**
   - **通俗理解**：第i个样本的"正确答案"或"类别"
   - **Intuitive**: "Correct answer" or "category" of i-th sample
   - **例子**：y^(1) = 1 表示第1个样本属于类别1
   - **Example**: y^(1) = 1 means 1st sample belongs to category 1

6. **$\theta$ - 模型参数 / Model Parameters**
   - **通俗理解**：模型的"设置"或"配置"
   - **Intuitive**: "Settings" or "configuration" of the model
   - **例子**：θ可能包含"类别0的概率是0.4"这样的信息
   - **Example**: θ might contain information like "probability of class 0 is 0.4"

7. **$P(x^{(i)}, y^{(i)} \mid \theta)$ - 联合概率 / Joint Probability**
   - **通俗理解**：在参数θ下，同时看到特征x^(i)和标签y^(i)的概率
   - **Intuitive**: Probability of seeing both features x^(i) and label y^(i) under parameters θ
   - **例子**：P(x=(1,2), y=1 | θ) = 0.15 表示在参数θ下，看到特征(1,2)且标签是1的概率是15%
   - **Example**: P(x=(1,2), y=1 | θ) = 0.15 means under θ, probability of seeing features (1,2) with label 1 is 15%

**🔰 零基础：为什么要相乘？/ Zero-Basics: Why Multiply?**

**直观理解 / Intuitive Understanding:**

想象你连续抛3次硬币，每次都是正面：
Imagine you flip a coin 3 times in a row, all are heads:

- 第1次正面概率 = 0.5
- 第2次正面概率 = 0.5（独立事件，不受第1次影响）
- 第3次正面概率 = 0.5（独立事件，不受前两次影响）

**3次都是正面的概率 = 0.5 × 0.5 × 0.5 = 0.125**

**为什么相乘？/ Why Multiply?**
- 因为每次抛硬币是"独立事件"（一次的结果不影响另一次）
- Because each coin flip is an "independent event" (one result doesn't affect another)
- 独立事件的联合概率 = 各个概率的乘积
- Joint probability of independent events = product of individual probabilities

**在机器学习中 / In Machine Learning:**
- 每个训练样本就像一次"抛硬币"
- Each training sample is like one "coin flip"
- 我们假设样本之间是独立的（一个样本不影响另一个）
- We assume samples are independent (one doesn't affect another)
- 所以所有样本同时出现的概率 = 各个样本概率的乘积
- So probability of all samples appearing = product of individual sample probabilities

**计算步骤 / Calculation Steps:**
1. 对每个训练样本i，计算 $P(x^{(i)}, y^{(i)} \mid \theta)$ / For each sample i, calculate $P(x^{(i)}, y^{(i)} \mid \theta)$
   - 这一步是计算"单个样本出现的概率"
   - This step calculates "probability of a single sample appearing"
2. 将所有概率相乘 / Multiply all probabilities together
   - 这一步是计算"所有样本同时出现的概率"
   - This step calculates "probability of all samples appearing together"

**计算示例 / Calculation Example:**

**步骤1：理解数据 / Step 1: Understand the Data**

**🔰 零基础：什么是"特征"和"标签"？/ Zero-Basics: What are "Features" and "Labels"?**

**生活例子 / Life Example:**
假设你要判断一个水果是"苹果"还是"橙子"：
Suppose you want to determine if a fruit is "apple" or "orange":

- **特征（Features）**：你能观察到的属性
  - **Features**: Attributes you can observe
  - 比如：颜色、大小、重量
  - E.g.: color, size, weight
- **标签（Label）**：正确答案（类别）
  - **Label**: Correct answer (category)
  - 比如：苹果=1，橙子=0
  - E.g.: apple=1, orange=0

**我们的例子 / Our Example:**
假设我们有一个二分类问题（y ∈ {0, 1}），每个样本有2个特征（x₁, x₂）
Suppose we have a binary classification problem (y ∈ {0, 1}), each sample has 2 features (x₁, x₂)

**训练数据集（就像你收集的样本）/ Training dataset (like samples you collected):**

| 样本编号 | 特征1 (x₁) | 特征2 (x₂) | 标签 (y) | 含义 |
| Sample # | Feature 1 (x₁) | Feature 2 (x₂) | Label (y) | Meaning |
|---------|------------|------------|---------|-------|
| 样本1 | 1 | 2 | 1 | 特征(1,2)对应类别1 |
| Sample 1 | 1 | 2 | 1 | Features (1,2) correspond to class 1 |
| 样本2 | 2 | 3 | 1 | 特征(2,3)对应类别1 |
| Sample 2 | 2 | 3 | 1 | Features (2,3) correspond to class 1 |
| 样本3 | 1 | 1 | 0 | 特征(1,1)对应类别0 |
| Sample 3 | 1 | 1 | 0 | Features (1,1) correspond to class 0 |

**用数学符号表示 / In Mathematical Notation:**
- 样本1 / Sample 1: x^(1) = (1, 2), y^(1) = 1
- 样本2 / Sample 2: x^(2) = (2, 3), y^(2) = 1  
- 样本3 / Sample 3: x^(3) = (1, 1), y^(3) = 0

**🔰 理解要点 / Key Points:**
- x^(1) 中的上标(1)表示"第1个样本"，不是"1次方"
- Superscript (1) in x^(1) means "1st sample", not "to the power of 1"
- (1, 2) 表示有2个特征，第一个特征值是1，第二个是2
- (1, 2) means 2 features, first feature value is 1, second is 2

**步骤2：理解朴素贝叶斯的假设 / Step 2: Understand Naive Bayes Assumptions**

**🔰 零基础：什么是"独立"？/ Zero-Basics: What is "Independent"?**

**生活例子 / Life Example:**
- **不独立**：今天下雨 → 明天也可能下雨（有关联）
  - **Not independent**: It rains today → It might rain tomorrow (related)
- **独立**：你抛硬币得到正面 → 不影响我抛硬币的结果（无关联）
  - **Independent**: You flip coin get heads → Doesn't affect my coin flip result (unrelated)

**在朴素贝叶斯中 / In Naive Bayes:**
- 我们假设**特征之间相互独立**
- We assume **features are independent of each other**
- 比如：特征1（颜色）和特征2（大小）互不影响
- E.g.: Feature 1 (color) and Feature 2 (size) don't affect each other

**为什么叫"朴素"？/ Why "Naive"?**
- 因为现实中特征往往有关联（比如"红色"和"圆形"在苹果中经常一起出现）
- Because in reality features are often related (e.g., "red" and "round" often appear together in apples)
- 但为了简化计算，我们"朴素地"假设它们独立
- But to simplify calculation, we "naively" assume they're independent

**数学公式 / Mathematical Formula:**

$$P(x, y \mid \theta) = P(y \mid \theta) \times P(x_1 \mid y, \theta) \times P(x_2 \mid y, \theta) \times ... \times P(x_n \mid y, \theta)$$

**通俗解释 / Intuitive Explanation:**
- 左边：看到特征x和标签y同时出现的概率
- Left side: Probability of seeing features x and label y together
- 右边：类别概率 × 特征1概率 × 特征2概率 × ...
- Right side: Class probability × Feature 1 probability × Feature 2 probability × ...

**为什么这样分解？/ Why This Decomposition?**
- 因为假设特征独立，所以联合概率 = 各个概率的乘积
- Because features are assumed independent, joint probability = product of individual probabilities

**符号说明 / Symbol Explanation:**
- $P(y \mid \theta)$：类别y的先验概率（"先验"=在观察特征之前就知道的）
  - **Prior probability** of class y ("prior" = known before observing features)
  - 比如：在不知道特征的情况下，一个样本属于类别1的概率是60%
  - E.g.: Without knowing features, probability of sample belonging to class 1 is 60%
- $P(x_j \mid y, \theta)$：在类别y下，第j个特征的条件概率
  - **Conditional probability** of j-th feature given class y
  - 比如：如果已知是类别1，那么特征1=1的概率是50%
  - E.g.: If known to be class 1, then probability of feature 1=1 is 50%
  
  **🔰 零基础：什么是"特征1=1"？/ Zero-Basics: What is "Feature 1=1"?**
  
  **详细解释 / Detailed Explanation:**
  - **"特征1"** = 第1个特征（第一个特征，不是"1次方"）
  - **"Feature 1"** = The 1st feature (first feature, not "to the power of 1")
  - **"=1"** = 这个特征的值为1
  - **"=1"** = The value of this feature is 1
  - **"特征1=1"** = 第1个特征的值为1
  - **"Feature 1=1"** = The 1st feature has value 1
  
  **具体例子 / Concrete Example:**
  假设一个样本有2个特征：
  Suppose a sample has 2 features:
  - 样本：x = (1, 2)
  - Sample: x = (1, 2)
  - 这里：特征1 = 1，特征2 = 2
  - Here: Feature 1 = 1, Feature 2 = 2
  
  **概率的含义 / Meaning of Probability:**
  - P(特征1=1 | y=1) = 0.5 的意思是：
  - P(feature 1=1 | y=1) = 0.5 means:
  - 在**类别1**的所有样本中，有50%的样本其**第1个特征的值是1**
  - Among all samples of **class 1**, 50% have **feature 1 with value 1**
  
  **用表格理解 / Understanding with Table:**
  
  | 样本编号 | 特征1的值 | 特征2的值 | 类别 |
  | Sample # | Feature 1 Value | Feature 2 Value | Class |
  |---------|-------------|-------------|------|
  | 样本1 | **1** | 2 | 1 |
  | Sample 1 | **1** | 2 | 1 |
  | 样本2 | **1** | 3 | 1 |
  | Sample 2 | **1** | 3 | 1 |
  | 样本3 | 2 | 2 | 1 |
  | Sample 3 | 2 | 2 | 1 |
  | 样本4 | 2 | 3 | 1 |
  | Sample 4 | 2 | 3 | 1 |
  
  - 在类别1的4个样本中，有2个样本的特征1=1（样本1和样本2）
  - Among 4 samples of class 1, 2 have feature 1=1 (sample 1 and sample 2)
  - 所以 P(特征1=1 | y=1) = 2/4 = 0.5 = 50%
  - So P(feature 1=1 | y=1) = 2/4 = 0.5 = 50%

**步骤3：假设已知的参数θ / Step 3: Assume Known Parameters θ**

**🔰 零基础：参数是什么？/ Zero-Basics: What are Parameters?**

**通俗理解 / Intuitive Understanding:**
参数就像"规则表"，告诉你在不同情况下概率是多少
Parameters are like "rule tables" that tell you probabilities in different situations

**类别先验概率 / Class Prior Probabilities:**
- **含义**：在不知道任何特征的情况下，一个样本属于各类别的概率
- **Meaning**: Probability of a sample belonging to each class without knowing any features
- P(y=0) = 0.4 → 40%的样本属于类别0
- P(y=0) = 0.4 → 40% of samples belong to class 0
- P(y=1) = 0.6 → 60%的样本属于类别1
- P(y=1) = 0.6 → 60% of samples belong to class 1
- **验证**：0.4 + 0.6 = 1.0 ✓（所有类别概率加起来等于1）
- **Check**: 0.4 + 0.6 = 1.0 ✓ (All class probabilities sum to 1)

**特征条件概率表 / Feature Conditional Probability Tables:**

**对于类别y=0 / For class y=0:**

| 特征 | 特征值 | 概率 | 含义 |
| Feature | Value | Probability | Meaning |
|------|------|-----------|-------|
| x₁ | 1 | 0.7 | 在类别0中，70%的样本特征1=1 |
| x₁ | 1 | 0.7 | In class 0, 70% of samples have feature 1=1 |
| x₁ | 2 | 0.3 | 在类别0中，30%的样本特征1=2 |
| x₁ | 2 | 0.3 | In class 0, 30% of samples have feature 1=2 |
| x₂ | 1 | 0.8 | 在类别0中，80%的样本特征2=1 |
| x₂ | 1 | 0.8 | In class 0, 80% of samples have feature 2=1 |
| x₂ | 2 | 0.2 | 在类别0中，20%的样本特征2=2 |
| x₂ | 2 | 0.2 | In class 0, 20% of samples have feature 2=2 |
| x₂ | 3 | 0.0 | 在类别0中，0%的样本特征2=3（不可能） |
| x₂ | 3 | 0.0 | In class 0, 0% of samples have feature 2=3 (impossible) |

**验证**：对于x₁，0.7 + 0.3 = 1.0 ✓；对于x₂，0.8 + 0.2 + 0.0 = 1.0 ✓
**Check**: For x₁, 0.7 + 0.3 = 1.0 ✓; For x₂, 0.8 + 0.2 + 0.0 = 1.0 ✓

**对于类别y=1 / For class y=1:**

| 特征 | 特征值 | 概率 | 含义 |
| Feature | Value | Probability | Meaning |
|------|------|-----------|-------|
| x₁ | 1 | 0.5 | 在类别1中，50%的样本特征1=1 |
| x₁ | 1 | 0.5 | In class 1, 50% of samples have feature 1=1 |
| x₁ | 2 | 0.5 | 在类别1中，50%的样本特征1=2 |
| x₁ | 2 | 0.5 | In class 1, 50% of samples have feature 1=2 |
| x₂ | 1 | 0.2 | 在类别1中，20%的样本特征2=1 |
| x₂ | 1 | 0.2 | In class 1, 20% of samples have feature 2=1 |
| x₂ | 2 | 0.5 | 在类别1中，50%的样本特征2=2 |
| x₂ | 2 | 0.5 | In class 1, 50% of samples have feature 2=2 |
| x₂ | 3 | 0.3 | 在类别1中，30%的样本特征2=3 |
| x₂ | 3 | 0.3 | In class 1, 30% of samples have feature 2=3 |

**验证**：对于x₁，0.5 + 0.5 = 1.0 ✓；对于x₂，0.2 + 0.5 + 0.3 = 1.0 ✓
**Check**: For x₁, 0.5 + 0.5 = 1.0 ✓; For x₂, 0.2 + 0.5 + 0.3 = 1.0 ✓

**🔰 这些参数从哪里来？/ Where Do These Parameters Come From?**
- 通常从训练数据中"学习"或"估计"得到
- Usually "learned" or "estimated" from training data
- 比如：如果训练数据中60%的样本是类别1，那么P(y=1) = 0.6
- E.g.: If 60% of training samples are class 1, then P(y=1) = 0.6
- 这里我们假设已经知道了这些参数（在实际应用中需要先估计）
- Here we assume we already know these parameters (in practice, we need to estimate them first)

**步骤4：计算每个样本的联合概率 / Step 4: Calculate Joint Probability for Each Sample**

**样本1: x=(1, 2), y=1**

**🔰 零基础：逐步计算 / Zero-Basics: Step-by-Step Calculation**

**步骤1：理解问题 / Step 1: Understand the Problem**
- 我们要计算：在参数θ下，看到特征(1,2)且标签是1的概率
- We want to calculate: Under parameters θ, probability of seeing features (1,2) with label 1

**步骤2：应用朴素贝叶斯公式 / Step 2: Apply Naive Bayes Formula**

根据朴素贝叶斯的独立假设：
According to Naive Bayes independence assumption:

$$P(x=(1,2), y=1 \mid \theta) = P(y=1) \times P(x_1=1 \mid y=1) \times P(x_2=2 \mid y=1)$$

**通俗解释 / Intuitive Explanation:**
- 这个公式说：要同时看到类别1、特征1=1、特征2=2，需要：
- This formula says: To see class 1, feature 1=1, and feature 2=2 together, we need:
  1. 首先是类别1（概率0.6）
  1. First be class 1 (probability 0.6)
  2. 在类别1中，特征1=1（概率0.5）
  2. In class 1, feature 1=1 (probability 0.5)
  3. 在类别1中，特征2=2（概率0.5）
  3. In class 1, feature 2=2 (probability 0.5)
- 因为假设独立，所以概率相乘
- Because of independence assumption, probabilities multiply

**步骤3：查找参数表 / Step 3: Look Up Parameter Tables**

从步骤3的参数表中查找：
Look up from parameter tables in Step 3:

- P(y=1) = 0.6（类别先验概率表）
- P(y=1) = 0.6 (from class prior probability table)
- P(x₁=1 | y=1) = 0.5（类别1的特征1条件概率表）
- P(x₁=1 | y=1) = 0.5 (from class 1's feature 1 conditional probability table)
- P(x₂=2 | y=1) = 0.5（类别1的特征2条件概率表）
- P(x₂=2 | y=1) = 0.5 (from class 1's feature 2 conditional probability table)

**步骤4：代入计算 / Step 4: Substitute and Calculate**

$$P(x=(1,2), y=1 \mid \theta) = 0.6 \times 0.5 \times 0.5$$

**详细计算过程 / Detailed Calculation Process:**
- 0.6 × 0.5 = 0.3（先算前两项）
- 0.6 × 0.5 = 0.3 (calculate first two terms)
- 0.3 × 0.5 = 0.15（再乘以第三项）
- 0.3 × 0.5 = 0.15 (multiply by third term)

**结果 / Result:**
$$P(x=(1,2), y=1 \mid \theta) = 0.15$$

**含义 / Meaning:**
- 在参数θ下，看到样本(特征1=1, 特征2=2, 标签=1)的概率是15%
- Under parameters θ, probability of seeing sample (feature 1=1, feature 2=2, label=1) is 15%

**样本2: x=(2, 3), y=1**
- $$P(x=(2,3), y=1 \mid \theta) = P(y=1) \times P(x_1=2 \mid y=1) \times P(x_2=3 \mid y=1)$$
- $$P(x=(2,3), y=1 \mid \theta) = 0.6 \times 0.5 \times 0.3 = 0.09$$

**样本3: x=(1, 1), y=0**
- $$P(x=(1,1), y=0 \mid \theta) = P(y=0) \times P(x_1=1 \mid y=0) \times P(x_2=1 \mid y=0)$$
- $$P(x=(1,1), y=0 \mid \theta) = 0.4 \times 0.7 \times 0.8 = 0.224$$

**步骤5：计算似然函数 / Step 5: Calculate Likelihood Function**

**🔰 零基础：什么是"所有样本同时出现"？/ Zero-Basics: What is "All Samples Appear Together"?**

**生活类比 / Life Analogy:**
- 你连续抛3次硬币，想知道"3次都是正面"的概率
- You flip a coin 3 times, want to know probability of "all 3 are heads"
- 这就是"所有事件同时发生"的概率
- This is probability of "all events happening together"

**在机器学习中 / In Machine Learning:**
- 我们想知道：在参数θ下，**同时看到这3个训练样本**的概率
- We want to know: Under parameters θ, probability of **seeing all 3 training samples together**
- 这就是"似然函数"的含义
- This is what "likelihood function" means

**数学公式 / Mathematical Formula:**

$$L(\theta) = \prod_{i=1}^{3} P(x^{(i)}, y^{(i)} \mid \theta)$$

**展开形式 / Expanded Form:**

$$L(\theta) = P(x^{(1)}, y^{(1)} \mid \theta) \times P(x^{(2)}, y^{(2)} \mid \theta) \times P(x^{(3)}, y^{(3)} \mid \theta)$$

**代入我们计算出的值 / Substitute Our Calculated Values:**

从前面步骤我们知道：
From previous steps we know:
- P(x^(1), y^(1)|θ) = 0.15（样本1的概率）
- P(x^(1), y^(1)|θ) = 0.15 (probability of sample 1)
- P(x^(2), y^(2)|θ) = 0.09（样本2的概率）
- P(x^(2), y^(2)|θ) = 0.09 (probability of sample 2)
- P(x^(3), y^(3)|θ) = 0.224（样本3的概率）
- P(x^(3), y^(3)|θ) = 0.224 (probability of sample 3)

**详细计算过程 / Detailed Calculation Process:**

$$L(\theta) = 0.15 \times 0.09 \times 0.224$$

**分步计算 / Step-by-Step:**
1. 先算前两项：0.15 × 0.09 = 0.0135
1. Calculate first two: 0.15 × 0.09 = 0.0135
2. 再乘以第三项：0.0135 × 0.224 = 0.003024
2. Multiply by third: 0.0135 × 0.224 = 0.003024

**最终结果 / Final Result:**

$$L(\theta) = 0.003024$$

**🔰 这个数字很小，正常吗？/ Is This Small Number Normal?**

**是的，完全正常！/ Yes, completely normal!**

**原因 / Reason:**
- 这是3个概率的乘积，每个概率都小于1
- This is product of 3 probabilities, each less than 1
- 多个小于1的数相乘，结果会越来越小
- Multiplying numbers less than 1 makes result smaller and smaller
- 比如：0.5 × 0.5 × 0.5 = 0.125（已经很小了）
- E.g.: 0.5 × 0.5 × 0.5 = 0.125 (already very small)

**实际意义 / Practical Meaning:**
- L(θ) = 0.003024 表示：在参数θ下，同时看到这3个特定样本的概率是0.3024%
- L(θ) = 0.003024 means: Under parameters θ, probability of seeing these 3 specific samples together is 0.3024%
- 这个概率很小是正常的，因为"恰好是这3个样本"是一个很具体的事件
- This small probability is normal, because "exactly these 3 samples" is a very specific event

**步骤6：解释结果 / Step 6: Interpret the Result**

**🔰 零基础：这个结果告诉我们什么？/ Zero-Basics: What Does This Result Tell Us?**

**1. 似然值的含义 / Meaning of Likelihood Value**
- L(θ) = 0.003024 表示：在参数θ下，观测到这3个样本的联合概率是0.3024%
- L(θ) = 0.003024 means: Under parameters θ, joint probability of observing these 3 samples is 0.3024%
- 换句话说：如果参数θ是正确的，那么看到这3个样本的可能性是0.3024%
- In other words: If parameters θ are correct, then possibility of seeing these 3 samples is 0.3024%

**2. 为什么值很小？/ Why Is the Value Small?**
- **完全正常！** 因为：
- **Completely normal!** Because:
  - 这是多个概率的乘积（每个都<1）
  - It's product of multiple probabilities (each <1)
  - 样本越多，乘积越小
  - More samples, smaller product
  - 如果有100个样本，结果可能是10^(-50)这样极小的数
  - If there are 100 samples, result might be extremely small like 10^(-50)

**3. 最大似然估计的目标 / Goal of Maximum Likelihood Estimation**

**核心思想 / Core Idea:**
- 我们想要找到**最好的参数θ**，使得L(θ)最大
- We want to find **best parameters θ** that maximize L(θ)
- 也就是说：找到让"看到这些数据"最可能的参数
- That is: Find parameters that make "seeing these data" most likely

**例子 / Example:**
假设我们尝试两组参数：
Suppose we try two sets of parameters:

| 参数组 | L(θ)值 | 评价 |
| Parameter Set | L(θ) Value | Evaluation |
|--------|---------|------|
| θ₁ | 0.003024 | 当前参数 |
| θ₁ | 0.003024 | Current parameters |
| θ₂ | 0.001000 | 更差（可能性更小） |
| θ₂ | 0.001000 | Worse (less likely) |
| θ₃ | 0.005000 | 更好（可能性更大） |
| θ₃ | 0.005000 | Better (more likely) |

- 如果L(θ₃) > L(θ₁)，说明θ₃比θ₁更好
- If L(θ₃) > L(θ₁), then θ₃ is better than θ₁
- 最大似然估计就是寻找使L(θ)最大的θ
- Maximum likelihood estimation seeks θ that maximizes L(θ)

**4. 实际应用 / Practical Application**

在实际中，我们通常：
In practice, we usually:
1. 从训练数据估计参数θ（比如计算各类别的频率）
1. Estimate parameters θ from training data (e.g., calculate frequencies of each class)
2. 计算似然函数L(θ)
2. Calculate likelihood function L(θ)
3. 调整参数使L(θ)最大（这就是"学习"过程）
3. Adjust parameters to maximize L(θ) (this is the "learning" process)

**对数似然 / Log-Likelihood:**  
$$
\ell(\theta) = \log L(\theta) = \sum_{i=1}^M \log P\big(x^{(i)}, y^{(i)} \mid \theta\big)
$$

**符号说明 / Symbol Explanation:**
- $\ell(\theta)$：对数似然函数 / Log-likelihood function
- $\log$：自然对数（或常用对数）/ Natural logarithm (or common logarithm)
- $\sum_{i=1}^M$：从i=1到M的求和符号 / Sum from i=1 to M

**计算步骤 / Calculation Steps:**
1. 对每个训练样本i，计算 $\log P(x^{(i)}, y^{(i)} \mid \theta)$ / For each sample i, calculate $\log P(x^{(i)}, y^{(i)} \mid \theta)$
2. 将所有对数概率相加 / Sum all log probabilities
3. 优点：将乘法变为加法，避免数值下溢 / Advantage: converts multiplication to addition, avoids numerical underflow

**计算示例 / Calculation Example:**
使用上面详细例子中的概率值：
Using probability values from the detailed example above:

**步骤1：计算每个样本的对数概率 / Step 1: Calculate Log Probability for Each Sample**

- log P(x^(1), y^(1)|θ) = log(0.15) ≈ -1.897
- log P(x^(2), y^(2)|θ) = log(0.09) ≈ -2.408
- log P(x^(3), y^(3)|θ) = log(0.224) ≈ -1.495

**步骤2：计算对数似然 / Step 2: Calculate Log-Likelihood**

对数似然是对数概率的和：
Log-likelihood is the sum of log probabilities:

$$\ell(\theta) = \sum_{i=1}^{3} \log P(x^{(i)}, y^{(i)} \mid \theta)$$

$$\ell(\theta) = \log(0.15) + \log(0.09) + \log(0.224)$$

$$\ell(\theta) = -1.897 + (-2.408) + (-1.495)$$

$$\ell(\theta) = -5.800$$

**步骤3：验证 / Step 3: Verification**

验证对数似然与似然函数的关系：
Verify relationship between log-likelihood and likelihood:

$$\ell(\theta) = \log L(\theta) = \log(0.003024) \approx -5.800$$

✓ 验证通过 / Verification passed

**步骤4：为什么使用对数似然？/ Step 4: Why Use Log-Likelihood?**

**优点1：数值稳定性 / Advantage 1: Numerical Stability**
- 当概率很小时（如0.0001），直接相乘可能下溢（计算机无法表示）
- When probabilities are very small (e.g., 0.0001), direct multiplication may underflow (computer cannot represent)
- 例如：0.0001 × 0.0001 × 0.0001 = 1e-12（可能下溢）
- Example: 0.0001 × 0.0001 × 0.0001 = 1e-12 (may underflow)
- 但对数相加：log(0.0001) + log(0.0001) + log(0.0001) = -9.21 - 9.21 - 9.21 = -27.63（稳定）
- But log addition: log(0.0001) + log(0.0001) + log(0.0001) = -9.21 - 9.21 - 9.21 = -27.63 (stable)

**优点2：计算简化 / Advantage 2: Computational Simplification**
- 乘法变为加法，计算更快
- Multiplication becomes addition, faster computation
- 求导更简单（对数和求导比乘积求导简单）
- Derivatives are simpler (logarithm derivatives are simpler than product derivatives)

**优点3：优化方便 / Advantage 3: Optimization Convenience**
- 最大化L(θ)等价于最大化ℓ(θ)（因为log是单调递增函数）
- Maximizing L(θ) is equivalent to maximizing ℓ(θ) (because log is monotonic increasing)
- 但通常我们最小化负对数似然 -ℓ(θ)（转换为最小化问题）
- But usually we minimize negative log-likelihood -ℓ(θ) (convert to minimization problem)

---

## 🔰 零基础常见问题解答 / Zero-Basics FAQ

### 问题1：为什么概率值这么小？/ Q1: Why Are Probability Values So Small?

**回答 / Answer:**
- **这是正常的！** 因为：
- **This is normal!** Because:
  1. 概率值本身就在0到1之间
   1. Probability values are between 0 and 1
  2. 多个概率相乘，结果会越来越小
   2. Multiplying probabilities makes result smaller
  3. 样本越多，乘积越小
   3. More samples, smaller product

**例子 / Example:**
- 抛硬币3次都是正面：0.5³ = 0.125
- Flip coin 3 times all heads: 0.5³ = 0.125
- 抛硬币10次都是正面：0.5¹⁰ ≈ 0.001（非常小！）
- Flip coin 10 times all heads: 0.5¹⁰ ≈ 0.001 (very small!)

**重要理解 / Important Understanding:**
- 概率小 ≠ 不可能
- Small probability ≠ impossible
- 概率小 = 这个特定组合很少见，但确实可能发生
- Small probability = This specific combination is rare, but can happen

### 问题2：什么是"条件概率"？/ Q2: What is "Conditional Probability"?

**通俗解释 / Intuitive Explanation:**
- **条件概率** = 在某个"条件"下，事件发生的概率
- **Conditional probability** = Probability of event happening under some "condition"

**生活例子 / Life Example:**
- P(下雨 | 阴天) = 在"阴天"这个条件下，下雨的概率
- P(rain | cloudy) = Probability of rain under "cloudy" condition
- 通常比P(下雨)大，因为阴天更容易下雨
- Usually larger than P(rain), because cloudy weather makes rain more likely

**在我们的例子中 / In Our Example:**
- P(x₁=1 | y=1) = 在"类别是1"的条件下，特征1=1的概率
- P(x₁=1 | y=1) = Probability of feature 1=1 under condition "class is 1"
- 这告诉我们：在类别1中，特征1=1有多常见
- This tells us: How common is feature 1=1 in class 1

### 问题3：这些参数是怎么来的？/ Q3: How Do We Get These Parameters?

**简单方法（频率估计）/ Simple Method (Frequency Estimation):**

**步骤1：估计类别先验概率 / Step 1: Estimate Class Prior Probabilities**
- 数一数训练数据中各类别有多少个样本
- Count how many samples of each class in training data
- P(y=1) = (类别1的样本数) / (总样本数)
- P(y=1) = (number of class 1 samples) / (total samples)

**例子 / Example:**
- 如果有100个样本，60个是类别1，40个是类别0
- If 100 samples, 60 are class 1, 40 are class 0
- 那么 P(y=1) = 60/100 = 0.6, P(y=0) = 40/100 = 0.4
- Then P(y=1) = 60/100 = 0.6, P(y=0) = 40/100 = 0.4

**步骤2：估计特征条件概率 / Step 2: Estimate Feature Conditional Probabilities**
- 对于每个类别，数一数每个特征值出现的次数
- For each class, count occurrences of each feature value
- P(x₁=1 | y=1) = (类别1中特征1=1的样本数) / (类别1的总样本数)
- P(x₁=1 | y=1) = (number of class 1 samples with feature 1=1) / (total class 1 samples)

**例子 / Example:**
- 在60个类别1的样本中，30个的特征1=1
- Among 60 class 1 samples, 30 have feature 1=1
- 那么 P(x₁=1 | y=1) = 30/60 = 0.5
- Then P(x₁=1 | y=1) = 30/60 = 0.5

### 问题4：为什么要用"乘积"而不是"求和"？/ Q4: Why Multiply Instead of Sum?

**关键理解 / Key Understanding:**
- 因为样本是**独立事件**
- Because samples are **independent events**
- 独立事件的联合概率 = 乘积
- Joint probability of independent events = product

**对比 / Comparison:**

| 情况 | 公式 | 例子 |
| Situation | Formula | Example |
|--------|------|------|
| 独立事件（抛硬币） | 乘积 | P(3次正面) = 0.5 × 0.5 × 0.5 |
| Independent events (coin flip) | Product | P(3 heads) = 0.5 × 0.5 × 0.5 |
| 互斥事件（要么A要么B） | 求和 | P(A或B) = P(A) + P(B) |
| Mutually exclusive (either A or B) | Sum | P(A or B) = P(A) + P(B) |

**在我们的例子中 / In Our Example:**
- 样本1出现不影响样本2出现（独立）
- Sample 1 appearing doesn't affect sample 2 appearing (independent)
- 所以用乘积：P(样本1) × P(样本2) × P(样本3)
- So use product: P(sample 1) × P(sample 2) × P(sample 3)

### 问题5：似然值小，说明参数不好吗？/ Q5: Small Likelihood Means Bad Parameters?

**不一定！/ Not necessarily!**

**重要理解 / Important Understanding:**
- 似然值的**绝对值**不重要
- **Absolute value** of likelihood doesn't matter
- 重要的是**相对大小**（与其他参数比较）
- What matters is **relative size** (compared to other parameters)

**例子 / Example:**
- θ₁的似然值：0.003024
- Likelihood of θ₁: 0.003024
- θ₂的似然值：0.001000
- Likelihood of θ₂: 0.001000
- 虽然都很小，但θ₁比θ₂好（因为0.003024 > 0.001000）
- Although both are small, θ₁ is better than θ₂ (because 0.003024 > 0.001000)

**类比 / Analogy:**
- 就像考试分数：60分和80分都不完美，但80分更好
- Like exam scores: 60 and 80 are both imperfect, but 80 is better
- 我们找的是"相对最好"的参数，不是"绝对完美"的参数
- We seek "relatively best" parameters, not "absolutely perfect" ones

### 问题6：如何理解"最大似然估计"？/ Q6: How to Understand "Maximum Likelihood Estimation"?

**通俗解释 / Intuitive Explanation:**
- 就像"猜谜游戏"：你看到一些线索（训练数据），要猜出"最可能"的答案（参数）
- Like a "guessing game": You see some clues (training data), guess "most likely" answer (parameters)

**步骤 / Steps:**
1. 尝试一组参数θ₁，计算L(θ₁)
1. Try parameter set θ₁, calculate L(θ₁)
2. 尝试另一组参数θ₂，计算L(θ₂)
2. Try another parameter set θ₂, calculate L(θ₂)
3. 比较：如果L(θ₂) > L(θ₁)，说明θ₂更好
3. Compare: If L(θ₂) > L(θ₁), then θ₂ is better
4. 继续尝试，找到使L(θ)最大的θ
4. Keep trying, find θ that maximizes L(θ)

**实际方法 / Practical Method:**
- 通常用数学方法（求导、梯度下降等）自动找到最优参数
- Usually use mathematical methods (derivatives, gradient descent, etc.) to automatically find optimal parameters
- 不需要手动尝试所有可能
- Don't need to manually try all possibilities

### 问题7：为什么要用对数？/ Q7: Why Use Logarithm?

**简单回答 / Simple Answer:**
- 因为概率相乘会变得很小，计算机可能无法精确表示（下溢）
- Because multiplying probabilities makes them very small, computer may not represent them accurately (underflow)
- 对数把"乘法"变成"加法"，更稳定
- Logarithm converts "multiplication" to "addition", more stable

**例子 / Example:**
- 直接计算：0.001 × 0.001 × 0.001 = 0.000000001（可能丢失精度）
- Direct calculation: 0.001 × 0.001 × 0.001 = 0.000000001 (may lose precision)
- 对数计算：log(0.001) + log(0.001) + log(0.001) = -6.908 - 6.908 - 6.908 = -20.724（稳定）
- Log calculation: log(0.001) + log(0.001) + log(0.001) = -6.908 - 6.908 - 6.908 = -20.724 (stable)

### 问题8：如何验证我的计算是否正确？/ Q8: How to Verify My Calculation?

**验证方法 / Verification Methods:**

1. **检查概率值范围 / Check Probability Range**
   - 所有概率应该在0到1之间
   - All probabilities should be between 0 and 1
   - 如果某个概率>1或<0，计算肯定错了
   - If any probability >1 or <0, calculation is definitely wrong

2. **检查概率和 / Check Probability Sum**
   - 对于每个特征，所有可能值的概率和应该=1
   - For each feature, sum of probabilities of all possible values should = 1
   - 比如：P(x₁=1|y=1) + P(x₁=2|y=1) = 0.5 + 0.5 = 1.0 ✓
   - E.g.: P(x₁=1|y=1) + P(x₁=2|y=1) = 0.5 + 0.5 = 1.0 ✓

3. **验证对数关系 / Verify Logarithm Relationship**
   - log(L(θ)) 应该等于 ℓ(θ)
   - log(L(θ)) should equal ℓ(θ)
   - 比如：log(0.003024) ≈ -5.800 = ℓ(θ) ✓
   - E.g.: log(0.003024) ≈ -5.800 = ℓ(θ) ✓

---

### 2.3 拉普拉斯分布 / Laplacian Distribution

**中文解释：**
拉普拉斯分布用于建模实值特征，其概率密度函数为：

**English Explanation:**
The Laplacian distribution is used to model real-valued features, with probability density function:

**概率密度函数 / PDF:**  
$$
p(x\mid \mu,\sigma) = \frac{1}{2\sigma}\exp\!\left(-\frac{|x-\mu|}{\sigma}\right)
$$

**符号说明 / Symbol Explanation:**
- $p(x\mid \mu,\sigma)$：在参数$\mu$和$\sigma$下，随机变量取值为x的概率密度 / Probability density of random variable taking value x given parameters $\mu$ and $\sigma$
- $x$：随机变量的取值（实数）/ Value of random variable (real number)
- $\mu$：位置参数，也是分布的中位数和众数 / Location parameter, also median and mode of distribution
- $\sigma$：尺度参数，控制分布的宽度（必须>0）/ Scale parameter, controls width of distribution (must be >0)
- $|x-\mu|$：x与$\mu$的绝对距离 / Absolute distance between x and $\mu$
- $\exp$：自然指数函数，$e$的幂次 / Natural exponential function, $e$ raised to power
- $\frac{1}{2\sigma}$：归一化系数，确保概率密度函数积分为1 / Normalization coefficient, ensures PDF integrates to 1

**计算步骤 / Calculation Steps:**
1. 计算 $|x-\mu|$（x与$\mu$的绝对距离）/ Calculate $|x-\mu|$ (absolute distance)
2. 计算 $\frac{|x-\mu|}{\sigma}$（归一化距离）/ Calculate $\frac{|x-\mu|}{\sigma}$ (normalized distance)
3. 计算 $-\frac{|x-\mu|}{\sigma}$（取负号）/ Calculate $-\frac{|x-\mu|}{\sigma}$ (take negative)
4. 计算 $\exp(-\frac{|x-\mu|}{\sigma})$（e的负归一化距离次方）/ Calculate $\exp(-\frac{|x-\mu|}{\sigma})$ (e to power of negative normalized distance)
5. 乘以归一化系数$\frac{1}{2\sigma}$ / Multiply by normalization coefficient $\frac{1}{2\sigma}$

**计算示例 / Calculation Example:**
假设参数μ=2, σ=1，计算x=3处的概率密度
Suppose parameters μ=2, σ=1, calculate probability density at x=3

步骤1: |3-2| = 1
步骤2: 1/1 = 1
步骤3: -1
步骤4: exp(-1) ≈ 0.368
步骤5: (1/(2×1)) × 0.368 = 0.5 × 0.368 = 0.184

所以 p(3|μ=2,σ=1) ≈ 0.184
So p(3|μ=2,σ=1) ≈ 0.184

再计算x=2（在μ处）:
Step 1: |2-2| = 0
Step 2: 0/1 = 0
Step 3: -0 = 0
Step 4: exp(0) = 1
Step 5: 0.5 × 1 = 0.5

在μ处概率密度最大（峰值）
Probability density is maximum at μ (peak)

**特点 / Characteristics:**
- 分布关于$\mu$对称 / Distribution is symmetric about $\mu$
- 在$\mu$处达到峰值 / Peak at $\mu$
- 尾部比高斯分布更厚（重尾分布）/ Heavier tails than Gaussian distribution

### 2.4 参数估计 / Parameter Estimation

**对于拉普拉斯分布 / For Laplacian Distribution:**

**位置参数 μ / Location Parameter μ:**
- 最大似然估计是中位数 / MLE is the median
- 因为拉普拉斯分布的中位数等于位置参数 / Because the median of Laplacian equals the location parameter

**尺度参数 σ / Scale Parameter σ:**
通过对数似然对 σ 求导并令其为零得到 / Obtained by taking derivative of log-likelihood w.r.t. σ and setting to zero

---

## 线性判别分析 / Linear Discriminant Analysis

### 3.1 基本概念 / Basic Concepts

**中文解释：**
线性判别分析（LDA）假设每个类别的数据服从高斯分布，通过贝叶斯定理计算后验概率。

**English Explanation:**
Linear Discriminant Analysis (LDA) assumes data from each class follows a Gaussian distribution and computes posterior probabilities using Bayes' theorem.

### 3.2 后验概率的Sigmoid形式 / Sigmoid Form of Posterior

**中文解释：**
在二分类问题中，LDA的后验概率可以写成sigmoid函数的形式。

**English Explanation:**
In binary classification, the posterior probability of LDA can be written in sigmoid form.

**推导过程 / Derivation:**

1. 使用贝叶斯定理 / Using Bayes' theorem:  
$$
p(y=1\mid x) = \frac{p(x\mid y=1)p(y=1)}{p(x\mid y=1)p(y=1) + p(x\mid y=0)p(y=0)}
$$

2. 假设高斯分布 / Assuming Gaussian distribution:  
$$
p(x\mid y=c) = \mathcal{N}(x;\,\mu_c,\Sigma)
$$

3. 经过代数变换得到 / After algebraic manipulation:  
$$
p(y=1\mid x) = \frac{1}{1+\exp(-\theta_0 - \theta^\top x)}
$$

其中 / where:  
$$
\theta_0 = \log\frac{p(y=1)}{p(y=0)} - \frac{1}{2}\big(\mu_1^\top\Sigma^{-1}\mu_1 - \mu_0^\top\Sigma^{-1}\mu_0\big),\quad
\theta = \Sigma^{-1}(\mu_1 - \mu_0)
$$

### 3.3 与逻辑回归的关系 / Relationship with Logistic Regression

**中文解释：**
LDA和逻辑回归都产生sigmoid形式的分类器，但假设不同：
- LDA假设高斯分布和共享协方差矩阵
- 逻辑回归不做分布假设

**English Explanation:**
Both LDA and logistic regression produce sigmoid classifiers, but with different assumptions:
- LDA assumes Gaussian distribution and shared covariance matrix
- Logistic regression makes no distributional assumptions

---

## KL散度 / KL Divergence

### 4.1 定义 / Definition

**中文解释：**
KL散度（Kullback-Leibler Divergence）衡量两个概率分布之间的差异。

**English Explanation:**
KL Divergence measures the difference between two probability distributions.

**定义 / Definition:**  
$$
D_{\mathrm{KL}}(P\|Q) = \int P(x)\,\log\frac{P(x)}{Q(x)}\,dx = \mathbb{E}_{P}\big[\log P(x)-\log Q(x)\big]
$$

**符号说明 / Symbol Explanation:**
- $D_{\mathrm{KL}}(P\|Q)$：从分布Q到分布P的KL散度（注意顺序！）/ KL divergence from distribution Q to distribution P (note the order!)
- $P(x)$：分布P在x处的概率密度 / Probability density of distribution P at x
- $Q(x)$：分布Q在x处的概率密度 / Probability density of distribution Q at x
- $\int$：积分符号，对所有可能的x值积分 / Integral symbol, integrate over all possible values of x
- $\log$：自然对数（或常用对数）/ Natural logarithm (or common logarithm)
- $\frac{P(x)}{Q(x)}$：两个分布的概率密度比值 / Ratio of probability densities of two distributions
- $\mathbb{E}_{P}[\cdot]$：在分布P下的期望值 / Expectation under distribution P
- $dx$：对x的积分 / Integration with respect to x

**计算步骤 / Calculation Steps:**
1. 对每个x值，计算 $\frac{P(x)}{Q(x)}$（两个分布的概率密度比）/ For each x, calculate $\frac{P(x)}{Q(x)}$ (ratio of probability densities)
2. 计算 $\log\frac{P(x)}{Q(x)} = \log P(x) - \log Q(x)$ / Calculate $\log\frac{P(x)}{Q(x)} = \log P(x) - \log Q(x)$
3. 乘以$P(x)$得到 $P(x) \log\frac{P(x)}{Q(x)}$ / Multiply by $P(x)$ to get $P(x) \log\frac{P(x)}{Q(x)}$
4. 对所有x积分（或求和，如果是离散分布）/ Integrate (or sum, if discrete) over all x
5. 结果总是≥0，当且仅当P=Q时等于0 / Result is always ≥0, equals 0 if and only if P=Q

**计算示例 / Calculation Example:**
假设两个离散分布：
Suppose two discrete distributions:

P: P(0)=0.5, P(1)=0.5
Q: Q(0)=0.8, Q(1)=0.2

计算KL散度 D_KL(P||Q) / Calculate KL divergence D_KL(P||Q):
- 当x=0: P(0)×log(P(0)/Q(0)) = 0.5×log(0.5/0.8) = 0.5×log(0.625) = 0.5×(-0.470) = -0.235
- 当x=1: P(1)×log(P(1)/Q(1)) = 0.5×log(0.5/0.2) = 0.5×log(2.5) = 0.5×0.916 = 0.458
- D_KL(P||Q) = -0.235 + 0.458 = 0.223

注意：D_KL(Q||P)会得到不同的值（非对称性）
Note: D_KL(Q||P) would give a different value (asymmetry)

**重要性质 / Important Properties:**
- 非对称：$D_{\mathrm{KL}}(P\|Q) \neq D_{\mathrm{KL}}(Q\|P)$ 通常 / Asymmetric: usually $D_{\mathrm{KL}}(P\|Q) \neq D_{\mathrm{KL}}(Q\|P)$
- 非负：$D_{\mathrm{KL}}(P\|Q) \geq 0$ / Non-negative: $D_{\mathrm{KL}}(P\|Q) \geq 0$
- 当P=Q时，KL散度为0 / When P=Q, KL divergence is 0

### 4.2 性质 / Properties

1. **非对称性 / Asymmetry:**
   - D_KL(P||Q) ≠ D_KL(Q||P) 通常 / in general

2. **非负性 / Non-negativity:**
   - D_KL(P||Q) ≥ 0
   - 当且仅当 P = Q 时等于0 / Equals 0 if and only if P = Q

3. **不是真正的距离 / Not a True Distance:**
   - 不满足三角不等式 / Does not satisfy triangle inequality

### 4.3 对称KL散度（Jeffreys散度）/ Symmetrized KL (Jeffreys Divergence)

**定义 / Definition:**  
$$
J(P_1, P_2) = D_{\mathrm{KL}}(P_1\|P_2) + D_{\mathrm{KL}}(P_2\|P_1)
$$

**中文解释：**
对称KL散度通过将两个方向的KL散度相加，得到一个对称的度量。

**English Explanation:**
Symmetrized KL divergence adds KL divergences in both directions to get a symmetric measure.

### 4.4 高斯分布的KL散度 / KL Divergence for Gaussian Distributions

**对于多元高斯分布 / For Multivariate Gaussian:**

两个N维高斯分布 P₁ = N(μ₁, Σ₁) 和 P₂ = N(μ₂, Σ₂) 的对称KL散度为：
For two N-dimensional Gaussians P₁ = N(μ₁, Σ₁) and P₂ = N(μ₂, Σ₂), the symmetrized KL divergence is:

$$
J(P_1, P_2) = \frac{1}{2}\text{tr}\big(\Sigma_1^{-1}\Sigma_2 + \Sigma_2^{-1}\Sigma_1 - 2I\big)
          + \frac{1}{2}(\mu_1-\mu_2)^\top(\Sigma_1^{-1}+\Sigma_2^{-1})(\mu_1-\mu_2)
$$

**关键技巧 / Key Techniques:**
- 使用迹的循环性质 / Using cyclic property of trace
- tr(AB) = tr(BA)
- tr((x-μ)ᵀΣ⁻¹(x-μ)) = tr(Σ⁻¹(x-μ)(x-μ)ᵀ)

---

## 学习建议 / Study Recommendations

### 对于初学者 / For Beginners:

1. **先掌握基础概率论 / Master Basic Probability First:**
   - 条件概率 / Conditional probability
   - 贝叶斯定理 / Bayes' theorem
   - 期望和方差 / Expectation and variance

2. **理解信息论概念 / Understand Information Theory:**
   - 从熵的直观理解开始 / Start with intuitive understanding of entropy
   - 信息量 = 不确定性 / Information = Uncertainty

3. **练习推导 / Practice Derivations:**
   - 不要只看答案 / Don't just read answers
   - 自己推导一遍 / Derive yourself
   - 理解每一步的数学原理 / Understand the math behind each step

4. **编程实现 / Programming Implementation:**
   - 实现朴素贝叶斯分类器 / Implement Naive Bayes classifier
   - 计算熵和互信息 / Compute entropy and mutual information
   - 可视化高斯分布 / Visualize Gaussian distributions

### 常见错误 / Common Mistakes:

1. **混淆联合概率和条件概率 / Confusing joint and conditional probability**
2. **忘记对数似然的负号 / Forgetting negative sign in log-likelihood**
3. **不理解指示函数的期望 / Not understanding expectation of indicator function**
4. **KL散度的方向混淆 / Confusing direction of KL divergence**

---

## 练习题 / Practice Problems

### 问题1 / Problem 1:
计算一个不公平硬币的熵，其中P(正面) = 0.7
Calculate the entropy of an unfair coin with P(heads) = 0.7

### 问题2 / Problem 2:
证明如果X和Y独立，则H(X, Y) = H(X) + H(Y)
Prove that if X and Y are independent, then H(X, Y) = H(X) + H(Y)

### 问题3 / Problem 3:
推导拉普拉斯分布的最大似然估计
Derive the maximum likelihood estimates for Laplacian distribution

---

## 参考资源 / Reference Resources

1. **概率论 / Probability:**
   - 《概率论与数理统计》/ "Probability and Mathematical Statistics"
   - Introduction to Probability (Blitzstein & Hwang)

2. **信息论 / Information Theory:**
   - Elements of Information Theory (Cover & Thomas)

3. **机器学习 / Machine Learning:**
   - Pattern Recognition and Machine Learning (Bishop)
   - Machine Learning: A Probabilistic Perspective (Murphy)

---

## 例题与解答 / Worked Examples

### 例题1：偏置骰子偶数概率 / Biased Die Even Probability

**题目 / Question:**  
一个偏置骰子六个面的概率分布如下：
A biased die has the following probabilities of landing on each face:

| 面 / Face | 1 | 2 | 3 | 4 | 5 | 6 |
|----------|---|---|---|---|---|---|
| 概率 P(face) | 0.1 | 0.1 | 0.2 | 0.2 | 0.4 | 0 |

如果掷出偶数就获胜，求获胜的概率。这个概率比公平骰子（每个面概率相等）更好还是更差？
I win if the die shows even. What is the probability that I win? Is this better or worse than a fair die (i.e., a die with equal probabilities for each face)?

**详细解答 / Detailed Solution:**

**步骤1：理解问题 / Step 1: Understand the Problem**
- 偶数面：2, 4, 6
- Even faces: 2, 4, 6
- 需要计算：P(2) + P(4) + P(6)
- Need to calculate: P(2) + P(4) + P(6)

**步骤2：计算概率 / Step 2: Calculate Probability**

$$P(\text{even}) = P(2) + P(4) + P(6)$$

从概率表中查找：
Look up from probability table:
- P(2) = 0.1
- P(4) = 0.2
- P(6) = 0

**步骤3：求和 / Step 3: Sum**

$$P(\text{even}) = 0.1 + 0.2 + 0 = 0.3$$

**步骤4：与公平骰子比较 / Step 4: Compare with Fair Die**

公平骰子每个面的概率都是1/6：
Fair die has probability 1/6 for each face:

$$P_{\text{fair}}(\text{even}) = P(2) + P(4) + P(6) = \frac{1}{6} + \frac{1}{6} + \frac{1}{6} = \frac{3}{6} = 0.5$$

**结论 / Conclusion:**
- 偏置骰子：P(even) = 0.3 = 30%
- Biased die: P(even) = 0.3 = 30%
- 公平骰子：P(even) = 0.5 = 50%
- Fair die: P(even) = 0.5 = 50%
- **偏置骰子更差**（获胜概率更低）
- **Biased die is worse** (lower winning probability)

**关键词 / Keywords:** 概率求和、事件并集、互斥事件。

### 例题2：指示函数期望 / Expectation of Indicator

**题目 / Question:**  
设随机变量X可以取值3、8或9，对应的概率分别为p₃、p₈和p₉。
Let X be a random variable which takes on the values 3, 8 or 9 with probabilities p₃, p₈ and p₉ respectively.

计算指示函数的期望值：E[I[X = 8]]
Calculate the expected value of the indicator function: E[I[X = 8]]

**详细解答 / Detailed Solution:**

**步骤1：理解指示函数 / Step 1: Understand Indicator Function**

指示函数的定义：
Definition of indicator function:

$$I[X = 8] = \begin{cases} 1, & \text{if } X = 8 \\ 0, & \text{otherwise} \end{cases}$$

**步骤2：应用期望的定义 / Step 2: Apply Definition of Expectation**

期望值的定义：
Definition of expected value:

$$E[I[X = 8]] = \sum_{x \in \{3,8,9\}} I[X = 8] \cdot P(X = x)$$

**步骤3：展开求和 / Step 3: Expand Sum**

$$E[I[X = 8]] = I[3 = 8] \cdot P(X = 3) + I[8 = 8] \cdot P(X = 8) + I[9 = 8] \cdot P(X = 9)$$

**步骤4：计算指示函数的值 / Step 4: Calculate Indicator Values**

- I[3 = 8] = 0（因为3 ≠ 8）
- I[3 = 8] = 0 (because 3 ≠ 8)
- I[8 = 8] = 1（因为8 = 8）
- I[8 = 8] = 1 (because 8 = 8)
- I[9 = 8] = 0（因为9 ≠ 8）
- I[9 = 8] = 0 (because 9 ≠ 8)

**步骤5：代入计算 / Step 5: Substitute and Calculate**

$$E[I[X = 8]] = 0 \cdot p_3 + 1 \cdot p_8 + 0 \cdot p_9 = p_8$$

**结论 / Conclusion:**
$$E[I[X = 8]] = p_8$$

**重要理解 / Important Understanding:**
- 指示函数的期望值等于该事件发生的概率
- Expected value of indicator function equals probability of that event
- 这是一个非常重要的性质，在概率论中经常使用
- This is a very important property, frequently used in probability theory

**关键词 / Keywords:** 指示函数、期望线性性、概率与期望的关系。

### 例题3：熵的链式法则 / Chain Rule of Entropy

**题目 / Question:**  
使用熵、联合熵和条件熵的定义，证明熵的链式法则：
Using the definitions of entropy, joint entropy, and conditional entropy, prove the chain rule for entropy:

$$H(X, Y) = H(Y) + H(X|Y)$$

**详细解答 / Detailed Solution:**

**步骤1：写出定义 / Step 1: Write Down Definitions**

**联合熵的定义 / Definition of Joint Entropy:**
$$H(X, Y) = -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} P(X=x, Y=y) \log_2 P(X=x, Y=y)$$

**条件熵的定义 / Definition of Conditional Entropy:**
$$H(X|Y) = -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} P(X=x, Y=y) \log_2 P(X=x|Y=y)$$

**熵的定义 / Definition of Entropy:**
$$H(Y) = -\sum_{y \in \mathcal{Y}} P(Y=y) \log_2 P(Y=y)$$

**步骤2：使用条件概率公式 / Step 2: Use Conditional Probability Formula**

根据条件概率的定义：
According to definition of conditional probability:

$$P(X=x, Y=y) = P(Y=y) \cdot P(X=x|Y=y)$$

**步骤3：展开联合熵 / Step 3: Expand Joint Entropy**

将条件概率公式代入联合熵：
Substitute conditional probability formula into joint entropy:

$$H(X, Y) = -\sum_{x,y} P(X=x, Y=y) \log_2 P(X=x, Y=y)$$

$$= -\sum_{x,y} P(Y=y) P(X=x|Y=y) \log_2 [P(Y=y) P(X=x|Y=y)]$$

**步骤4：使用对数性质分解 / Step 4: Decompose Using Logarithm Properties**

使用对数的乘积性质：$\log(ab) = \log a + \log b$
Use logarithm product property: $\log(ab) = \log a + \log b$

$$= -\sum_{x,y} P(Y=y) P(X=x|Y=y) [\log_2 P(Y=y) + \log_2 P(X=x|Y=y)]$$

**步骤5：展开并分离项 / Step 5: Expand and Separate Terms**

$$= -\sum_{x,y} P(Y=y) P(X=x|Y=y) \log_2 P(Y=y) - \sum_{x,y} P(Y=y) P(X=x|Y=y) \log_2 P(X=x|Y=y)$$

**步骤6：简化第一项 / Step 6: Simplify First Term**

对x求和，利用 $\sum_x P(X=x|Y=y) = 1$：
Sum over x, using $\sum_x P(X=x|Y=y) = 1$:

$$-\sum_{x,y} P(Y=y) P(X=x|Y=y) \log_2 P(Y=y) = -\sum_y P(Y=y) \log_2 P(Y=y) = H(Y)$$

**步骤7：简化第二项 / Step 7: Simplify Second Term**

第二项就是条件熵的定义：
Second term is the definition of conditional entropy:

$$-\sum_{x,y} P(Y=y) P(X=x|Y=y) \log_2 P(X=x|Y=y) = H(X|Y)$$

**步骤8：得出结论 / Step 8: Conclude**

$$H(X, Y) = H(Y) + H(X|Y) \quad \square$$

**验证 / Verification:**
同样可以证明：$H(X, Y) = H(X) + H(Y|X)$
Similarly we can prove: $H(X, Y) = H(X) + H(Y|X)$

**关键词 / Keywords:** 条件概率、对数分解、链式法则、联合熵。

### 例题4：独立随机变量的互信息 / Mutual Information of Independent Variables

**题目 / Question:**  
回忆两个随机变量X和Y独立的定义：对于所有x ∈ X和所有y ∈ Y，有
Recall that two random variables X and Y are independent if for all x ∈ X and all y ∈ Y:

$$P(X=x, Y=y) = P(X=x) P(Y=y)$$

如果变量X和Y独立，那么I(X; Y) = 0吗？如果是，请证明；如果不是，请给出反例。
If variables X and Y are independent, is I(X; Y) = 0? If yes, prove it. If no, give a counterexample.

**详细解答 / Detailed Solution:**

**答案：是的，I(X; Y) = 0 / Answer: Yes, I(X; Y) = 0**

**步骤1：写出互信息的定义 / Step 1: Write Definition of Mutual Information**

$$I(X; Y) = \sum_{x,y} P(X=x, Y=y) \log_2 \frac{P(X=x, Y=y)}{P(X=x) P(Y=y)}$$

**步骤2：使用独立性条件 / Step 2: Use Independence Condition**

由于X和Y独立：
Since X and Y are independent:

$$P(X=x, Y=y) = P(X=x) P(Y=y)$$

**步骤3：代入互信息公式 / Step 3: Substitute into Mutual Information Formula**

$$I(X; Y) = \sum_{x,y} P(X=x) P(Y=y) \log_2 \frac{P(X=x) P(Y=y)}{P(X=x) P(Y=y)}$$

**步骤4：简化 / Step 4: Simplify**

$$\frac{P(X=x) P(Y=y)}{P(X=x) P(Y=y)} = 1$$

所以：
Therefore:

$$I(X; Y) = \sum_{x,y} P(X=x) P(Y=y) \log_2 1$$

**步骤5：计算对数 / Step 5: Calculate Logarithm**

$$\log_2 1 = 0$$

**步骤6：得出结论 / Step 6: Conclude**

$$I(X; Y) = \sum_{x,y} P(X=x) P(Y=y) \cdot 0 = 0 \quad \square$$

**重要理解 / Important Understanding:**
- 互信息衡量两个变量的相互依赖程度
- Mutual information measures mutual dependence between two variables
- 如果两个变量独立，它们之间没有信息共享，所以互信息为0
- If two variables are independent, they share no information, so mutual information is 0
- 这是互信息的一个重要性质
- This is an important property of mutual information

**关键词 / Keywords:** 独立性、互信息、条件概率、对数性质。

---

### 例题5：拉普拉斯分布的最大似然估计 / MLE for Laplacian Distribution

**题目 / Question:**  
给定训练集 $D = \{(x^{(i)}, y^{(i)}); i = 1, \ldots, M\}$，其中 $x^{(i)} \in \mathbb{R}^N$ 且 $y^{(i)} \in \{1, 2, \ldots, C\}$，推导朴素贝叶斯对实值 $x_j^{(i)}$ 使用拉普拉斯分布建模时的最大似然估计。
Given a training set $D = \{(x^{(i)}, y^{(i)}); i = 1, \ldots, M\}$, where $x^{(i)} \in \mathbb{R}^N$ and $y^{(i)} \in \{1, 2, \ldots, C\}$, derive the maximum likelihood estimates of naive Bayes for real valued $x_j^{(i)}$ modeled with a Laplacian distribution.

**详细解答 / Detailed Solution:**

**步骤1：写出似然函数 / Step 1: Write Likelihood Function**

给定训练集，数据的联合概率分布为：
Given training set, joint probability distribution of data:

$$L(\phi, \theta) = \prod_{i=1}^M P(x^{(i)}, y^{(i)} | \phi, \theta)$$

其中φ是类别先验参数，θ是特征分布参数。
where φ are class prior parameters, θ are feature distribution parameters.

**步骤2：使用对数似然 / Step 2: Use Log-Likelihood**

对数似然函数：
Log-likelihood function:

$$\ell(\phi, \theta) = \sum_{i=1}^M \log P(x^{(i)}, y^{(i)} | \phi, \theta)$$

**步骤3：拉普拉斯分布模型 / Step 3: Laplacian Distribution Model**

对于实值特征 $x_j$，使用拉普拉斯分布建模：
For real-valued feature $x_j$, model with Laplacian distribution:

$$p(x_j | \mu_{jc}, \sigma_{jc}) = \frac{1}{2\sigma_{jc}} \exp\left(-\frac{|x_j - \mu_{jc}|}{\sigma_{jc}}\right)$$

其中 $\mu_{jc}$ 是类别c下特征j的位置参数，$\sigma_{jc}$ 是尺度参数。
where $\mu_{jc}$ is location parameter for feature j in class c, $\sigma_{jc}$ is scale parameter.

**步骤4：提取相关项 / Step 4: Extract Relevant Terms**

从对数似然中提取只依赖于 $\mu_{jc}$ 和 $\sigma_{jc}$ 的项：
Extract terms from log-likelihood that depend only on $\mu_{jc}$ and $\sigma_{jc}$:

$$\ell(\mu_{jc}, \sigma_{jc}) = \sum_{i:y^{(i)}=c} \left[-\log(2\sigma_{jc}) - \frac{|x_j^{(i)} - \mu_{jc}|}{\sigma_{jc}}\right] + \text{常数项}$$

**步骤5：估计位置参数μ / Step 5: Estimate Location Parameter μ**

对于拉普拉斯分布，位置参数的最大似然估计是中位数：
For Laplacian distribution, MLE of location parameter is the median:

$$\mu_{jc}^* = \text{median}\{x_j^{(i)} : y^{(i)} = c\}$$

**原因 / Reason:**
- 拉普拉斯分布的中位数等于位置参数
- Median of Laplacian distribution equals location parameter
- 当μ是中位数时，对μ的导数在大多数点为零
- When μ is median, derivative w.r.t. μ is zero at most points

**步骤6：估计尺度参数σ / Step 6: Estimate Scale Parameter σ**

对 $\sigma_{jc}$ 求导并令其为零：
Take derivative w.r.t. $\sigma_{jc}$ and set to zero:

$$\frac{\partial \ell}{\partial \sigma_{jc}} = -\frac{M_c}{\sigma_{jc}} + \frac{1}{\sigma_{jc}^2} \sum_{i:y^{(i)}=c} |x_j^{(i)} - \mu_{jc}| = 0$$

其中 $M_c$ 是类别c的样本数。
where $M_c$ is number of samples in class c.

**求解 / Solve:**

$$\sigma_{jc}^* = \frac{1}{M_c} \sum_{i:y^{(i)}=c} |x_j^{(i)} - \mu_{jc}^*|$$

**结论 / Conclusion:**
- 位置参数：$\mu_{jc}^*$ = 类别c中特征j的中位数
- Location parameter: $\mu_{jc}^*$ = median of feature j in class c
- 尺度参数：$\sigma_{jc}^*$ = 类别c中特征j的平均绝对偏差
- Scale parameter: $\sigma_{jc}^*$ = mean absolute deviation of feature j in class c

**关键词 / Keywords:** 最大似然估计、拉普拉斯分布、中位数、平均绝对偏差。

---

### 例题6：LDA后验概率的Sigmoid形式 / Sigmoid Form of LDA Posterior

**题目 / Question:**  
证明在二分类问题中，线性判别分析的后验概率 $p(y=1|x; \phi, \mu, \Sigma)$ 可以写成sigmoid形式：
Prove that in binary classification, the posterior of linear discriminant analysis, i.e., $p(y=1|x; \phi, \mu, \Sigma)$, admits a sigmoid form:

$$p(y=1|x) = \frac{1}{1+\exp(-\theta_0 - \theta^\top x)}$$

其中θ是$\{\phi, \mu, \Sigma\}$的函数。提示：记住使用约定 $x_0 = 1$。
where θ is a function of $\{\phi, \mu, \Sigma\}$. Hint: remember to use the convention of letting $x_0 = 1$.

**详细解答 / Detailed Solution:**

**步骤1：使用贝叶斯定理 / Step 1: Use Bayes' Theorem**

$$p(y=1|x) = \frac{p(x|y=1)p(y=1)}{p(x|y=1)p(y=1) + p(x|y=0)p(y=0)}$$

**步骤2：假设高斯分布 / Step 2: Assume Gaussian Distribution**

假设每个类别的数据服从高斯分布：
Assume data from each class follows Gaussian distribution:

$$p(x|y=c) = \mathcal{N}(x; \mu_c, \Sigma)$$

其中两个类别共享相同的协方差矩阵Σ。
where both classes share the same covariance matrix Σ.

**步骤3：代入高斯分布 / Step 3: Substitute Gaussian Distribution**

$$p(y=1|x) = \frac{\mathcal{N}(x; \mu_1, \Sigma) \phi_1}{\mathcal{N}(x; \mu_1, \Sigma) \phi_1 + \mathcal{N}(x; \mu_0, \Sigma) \phi_0}$$

其中 $\phi_1 = p(y=1)$, $\phi_0 = p(y=0)$。
where $\phi_1 = p(y=1)$, $\phi_0 = p(y=0)$.

**步骤4：展开高斯分布 / Step 4: Expand Gaussian Distribution**

多元高斯分布的概率密度函数：
Probability density function of multivariate Gaussian:

$$\mathcal{N}(x; \mu, \Sigma) = \frac{1}{(2\pi)^{N/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu)\right)$$

**步骤5：代数变换 / Step 5: Algebraic Manipulation**

经过代数变换（详细推导见课程笔记），得到：
After algebraic manipulation (detailed derivation in lecture notes):

$$p(y=1|x) = \frac{1}{1 + \exp(-\theta_0 - \theta^\top x)}$$

其中：
where:

$$\theta_0 = \log\frac{\phi_1}{\phi_0} - \frac{1}{2}(\mu_1^\top\Sigma^{-1}\mu_1 - \mu_0^\top\Sigma^{-1}\mu_0)$$

$$\theta = \Sigma^{-1}(\mu_1 - \mu_0)$$

**步骤6：使用约定 $x_0 = 1$ / Step 6: Use Convention $x_0 = 1$**

如果我们将 $x_0 = 1$ 包含在特征向量中，那么：
If we include $x_0 = 1$ in the feature vector, then:

$$\theta^\top x = \theta_0 x_0 + \theta_1 x_1 + \ldots + \theta_N x_N = \theta_0 + \sum_{i=1}^N \theta_i x_i$$

这样可以将偏置项 $\theta_0$ 整合到权重向量中。
This allows us to incorporate the bias term $\theta_0$ into the weight vector.

**结论 / Conclusion:**
LDA的后验概率确实可以写成sigmoid形式，这与逻辑回归的形式相同，但LDA有更强的分布假设。
LDA posterior can indeed be written in sigmoid form, same as logistic regression, but LDA has stronger distributional assumptions.

**关键词 / Keywords:** 贝叶斯定理、高斯分布、sigmoid函数、线性判别分析。

---

### 例题7：高斯分布的对称KL散度 / Symmetrized KL for Gaussians

**题目 / Question:**  
两个N维多元高斯分布 $P_1 = \mathcal{N}(x; \mu_1, \Sigma_1)$ 和 $P_2 = \mathcal{N}(x; \mu_2, \Sigma_2)$ 的对称KL散度（Jeffreys散度）定义为：
The symmetrized KL divergence (Jeffreys divergence) between two N-dimensional multivariate Gaussian distributions $P_1 = \mathcal{N}(x; \mu_1, \Sigma_1)$ and $P_2 = \mathcal{N}(x; \mu_2, \Sigma_2)$ is defined as:

$$J(P_1, P_2) = D_{\text{KL}}(P_1\|P_2) + D_{\text{KL}}(P_2\|P_1)$$

证明 $J(P_1, P_2)$ 可以写成闭式形式：
Prove that $J(P_1, P_2)$ can be written in closed form as:

$$J(P_1, P_2) = \frac{1}{2}\text{tr}(\Sigma_1^{-1}\Sigma_2 + \Sigma_2^{-1}\Sigma_1 - 2I) + \frac{1}{2}(\mu_1-\mu_2)^\top(\Sigma_1^{-1}+\Sigma_2^{-1})(\mu_1-\mu_2)$$

**详细解答 / Detailed Solution:**

**步骤1：写出KL散度的定义 / Step 1: Write Definition of KL Divergence**

$$D_{\text{KL}}(P_1\|P_2) = \mathbb{E}_{P_1}[\log P_1 - \log P_2]$$

**步骤2：展开高斯分布的对数 / Step 2: Expand Logarithm of Gaussian**

对于多元高斯分布：
For multivariate Gaussian:

$$\log \mathcal{N}(x; \mu, \Sigma) = -\frac{N}{2}\log(2\pi) - \frac{1}{2}\log|\Sigma| - \frac{1}{2}(x-\mu)^\top\Sigma^{-1}(x-\mu)$$

**步骤3：计算KL散度 / Step 3: Calculate KL Divergence**

$$D_{\text{KL}}(P_1\|P_2) = \mathbb{E}_{P_1}\left[\log P_1 - \log P_2\right]$$

$$= \mathbb{E}_{P_1}\left[-\frac{1}{2}\log|\Sigma_1| - \frac{1}{2}(x-\mu_1)^\top\Sigma_1^{-1}(x-\mu_1) + \frac{1}{2}\log|\Sigma_2| + \frac{1}{2}(x-\mu_2)^\top\Sigma_2^{-1}(x-\mu_2)\right]$$

**步骤4：使用期望的性质 / Step 4: Use Properties of Expectation**

对于 $x \sim \mathcal{N}(\mu_1, \Sigma_1)$：
For $x \sim \mathcal{N}(\mu_1, \Sigma_1)$:

$$\mathbb{E}[(x-\mu_1)^\top\Sigma_1^{-1}(x-\mu_1)] = \text{tr}(\Sigma_1^{-1}\Sigma_1) = \text{tr}(I) = N$$

$$\mathbb{E}[(x-\mu_2)^\top\Sigma_2^{-1}(x-\mu_2)] = \text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_1-\mu_2)^\top\Sigma_2^{-1}(\mu_1-\mu_2)$$

**步骤5：计算对称KL散度 / Step 5: Calculate Symmetrized KL**

$$J(P_1, P_2) = D_{\text{KL}}(P_1\|P_2) + D_{\text{KL}}(P_2\|P_1)$$

经过详细计算（使用迹的循环性质）：
After detailed calculation (using cyclic property of trace):

$$J(P_1, P_2) = \frac{1}{2}\text{tr}(\Sigma_1^{-1}\Sigma_2 + \Sigma_2^{-1}\Sigma_1 - 2I) + \frac{1}{2}(\mu_1-\mu_2)^\top(\Sigma_1^{-1}+\Sigma_2^{-1})(\mu_1-\mu_2)$$

**关键技巧 / Key Techniques:**
- 使用迹的循环性质：$\text{tr}(AB) = \text{tr}(BA)$
- Use cyclic property of trace: $\text{tr}(AB) = \text{tr}(BA)$
- 使用二次型的期望公式
- Use expectation formula for quadratic forms
- 对称KL散度消除了KL散度的非对称性
- Symmetrized KL eliminates asymmetry of KL divergence

**关键词 / Keywords:** KL散度、高斯分布、迹运算、对称散度。

