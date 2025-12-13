# Lecture 8 知识点总结 / Knowledge Summary

## Table of Contents / 目录

- [1. Neural Networks Introduction / 1. 神经网络导论](#1-neural-networks-introduction--1-神经网络导论)
- [2. Perceptron / 2. 感知机](#2-perceptron--2-感知机)
- [3. Multi-layer Perceptron / 3. 多层感知机](#3-multi-layer-perceptron--3-多层感知机)
- [4. Backpropagation / 4. 反向传播](#4-backpropagation--4-反向传播)

---

## 1. Neural Networks Introduction / 1. 神经网络导论

#### English
Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that can learn complex patterns from data.

#### 中文
神经网络是受生物神经网络启发的计算模型。它们由组织成层的互连节点（神经元）组成，可以从数据中学习复杂模式。

**基本结构 / Basic Structure:**
- **输入层 / Input Layer**: 接收输入特征 / Receives input features
- **隐藏层 / Hidden Layers**: 中间处理层 / Intermediate processing layers
- **输出层 / Output Layer**: 产生最终预测 / Produces final predictions

**神经元 / Neuron:**
每个神经元计算加权输入和，然后应用激活函数 / Each neuron computes weighted sum of inputs, then applies activation function

**数学表示 / Mathematical Representation:**
$$
z = \sum_{i=1}^n w_i x_i + b
$$

$$
a = \sigma(z)
$$

其中 / where:
- $x_i$：输入 / Inputs
- $w_i$：权重 / Weights
- $b$：偏置 / Bias
- $\sigma$：激活函数 / Activation function
- $a$：输出（激活值）/ Output (activation)

#### 通俗解释
神经网络就像"多层决策"：输入层接收数据，隐藏层逐步处理，输出层给出答案。每个神经元像一个小决策器，把所有输入加权求和，然后决定输出什么。

---

## 2. Perceptron / 2. 感知机

#### English
The perceptron is the simplest neural network, consisting of a single neuron. It can learn linear decision boundaries for binary classification.

#### 中文
感知机是最简单的神经网络，由单个神经元组成。它可以学习二元分类的线性决策边界。

**感知机模型 / Perceptron Model:**
$$
f(\mathbf{x}) = \text{sign}(\mathbf{w}^\top \mathbf{x} + b)
$$

其中 / where:
- $\text{sign}(z) = \begin{cases} +1 & z \ge 0 \\ -1 & z < 0 \end{cases}$：符号函数 / Sign function

**学习算法 / Learning Algorithm:**

1. 初始化权重和偏置 / Initialize weights and bias
2. 对每个训练样本 / For each training sample:
   - 计算预测 / Compute prediction: $\hat{y} = \text{sign}(\mathbf{w}^\top \mathbf{x} + b)$
   - 如果预测错误 / If prediction is wrong:
     $$
     \mathbf{w} \leftarrow \mathbf{w} + \eta y \mathbf{x}
     $$
     $$
     b \leftarrow b + \eta y
     $$
   其中$\eta$是学习率 / where $\eta$ is learning rate

3. 重复直到收敛或达到最大迭代次数 / Repeat until convergence or max iterations

**收敛性 / Convergence:**
如果数据线性可分，感知机保证收敛 / If data is linearly separable, perceptron is guaranteed to converge

#### 通俗解释
感知机就像"最简单的分类器"：它找一条直线把两类分开。如果分错了，就调整直线的位置，直到分对为止。但只能处理线性可分的数据。

---

## 3. Multi-layer Perceptron / 3. 多层感知机

#### English
Multi-layer Perceptron (MLP) extends the perceptron by adding hidden layers, enabling it to learn nonlinear decision boundaries and complex patterns.

#### 中文
多层感知机（MLP）通过添加隐藏层扩展感知机，使其能够学习非线性决策边界和复杂模式。

**网络结构 / Network Structure:**

**前向传播 / Forward Propagation:**

对于第$l$层 / For layer $l$:
$$
\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}
$$

$$
\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})
$$

其中 / where:
- $\mathbf{W}^{(l)}$：第$l$层的权重矩阵 / Weight matrix for layer $l$
- $\mathbf{b}^{(l)}$：第$l$层的偏置向量 / Bias vector for layer $l$
- $\mathbf{a}^{(l)}$：第$l$层的激活值 / Activations for layer $l$

**激活函数 / Activation Functions:**

1. **Sigmoid:**
   $$
   \sigma(z) = \frac{1}{1 + e^{-z}}
   $$

2. **Tanh:**
   $$
   \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
   $$

3. **ReLU:**
   $$
   \text{ReLU}(z) = \max(0, z)
   $$

**通用逼近定理 / Universal Approximation Theorem:**
具有一个隐藏层的MLP可以逼近任何连续函数（在有限域上）/ MLP with one hidden layer can approximate any continuous function (on finite domain)

#### 通俗解释
多层感知机就像"多层决策"：第一层做简单判断，第二层组合这些判断，最终得到复杂决策。就像多个专家讨论：每个专家（神经元）给出意见，最后综合所有意见做决定。

---

## 4. Backpropagation / 4. 反向传播

#### English
Backpropagation is an algorithm for training neural networks by computing gradients of the loss function with respect to weights using the chain rule of calculus.

#### 中文
反向传播是一种训练神经网络的算法，通过使用微积分的链式法则计算损失函数对权重的梯度。

**算法步骤 / Algorithm Steps:**

1. **前向传播 / Forward Pass:**
   计算所有层的激活值 / Compute activations for all layers

2. **计算损失 / Compute Loss:**
   $$
   J = \frac{1}{M}\sum_{i=1}^M L(y^{(i)}, \hat{y}^{(i)})
   $$

3. **反向传播 / Backward Pass:**
   从输出层向输入层传播误差 / Propagate error from output layer to input layer

**梯度计算 / Gradient Calculation:**

**输出层梯度 / Output Layer Gradient:**
$$
\delta^{(L)} = \frac{\partial J}{\partial \mathbf{a}^{(L)}} \odot \sigma'(\mathbf{z}^{(L)})
$$

**隐藏层梯度 / Hidden Layer Gradient:**
$$
\delta^{(l)} = ((\mathbf{W}^{(l+1)})^\top \delta^{(l+1)}) \odot \sigma'(\mathbf{z}^{(l)})
$$

**权重梯度 / Weight Gradients:**
$$
\frac{\partial J}{\partial \mathbf{W}^{(l)}} = \delta^{(l)} (\mathbf{a}^{(l-1)})^\top
$$

$$
\frac{\partial J}{\partial \mathbf{b}^{(l)}} = \delta^{(l)}
$$

**参数更新 / Parameter Update:**
$$
\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \alpha \frac{\partial J}{\partial \mathbf{W}^{(l)}}
$$

$$
\mathbf{b}^{(l)} \leftarrow \mathbf{b}^{(l)} - \alpha \frac{\partial J}{\partial \mathbf{b}^{(l)}}
$$

其中$\alpha$是学习率 / where $\alpha$ is learning rate

#### 通俗解释
反向传播就像"倒推错误"：从输出层的错误开始，一层层往前推，找出每层应该承担多少责任（梯度），然后根据责任调整权重。就像追责：最终错误了，往前找谁的责任，然后改正。

---

## Additional Detail / 补充要点

### Loss Functions / 损失函数

#### English
- **Mean Squared Error (MSE)**: For regression
  $$
  L(y, \hat{y}) = (y - \hat{y})^2
  $$
- **Cross-Entropy**: For classification
  $$
  L(y, \hat{y}) = -y\log(\hat{y}) - (1-y)\log(1-\hat{y})
  $$

#### 中文
- **均方误差（MSE）**：用于回归
  $$
  L(y, \hat{y}) = (y - \hat{y})^2
  $$
- **交叉熵**：用于分类
  $$
  L(y, \hat{y}) = -y\log(\hat{y}) - (1-y)\log(1-\hat{y})
  $$

---

### Regularization / 正则化

#### English
- **L2 Regularization**: Add $\frac{\lambda}{2}\|\mathbf{W}\|^2$ to loss
- **Dropout**: Randomly set some activations to zero during training
- **Early Stopping**: Stop training when validation error increases

#### 中文
- **L2正则化**：向损失添加$\frac{\lambda}{2}\|\mathbf{W}\|^2$
- **Dropout**：训练时随机将一些激活值设为零
- **早停**：验证误差增加时停止训练

---

### Optimization / 优化

#### English
- **Stochastic Gradient Descent (SGD)**: Update using one sample at a time
- **Mini-batch Gradient Descent**: Update using small batches
- **Momentum**: Add velocity term to smooth updates
- **Adam**: Adaptive learning rate method

#### 中文
- **随机梯度下降（SGD）**：每次用一个样本更新
- **小批量梯度下降**：用小批量更新
- **动量**：添加速度项平滑更新
- **Adam**：自适应学习率方法

---

## Learning Recommendations / 学习建议

### For Beginners / 对于初学者

1. **Understand Forward Propagation / 理解前向传播:**
   - Trace through a simple network
   - Compute activations manually
   - See how information flows

2. **Understand Backpropagation / 理解反向传播:**
   - Derive gradients for simple network
   - Implement from scratch
   - Visualize gradient flow

3. **Experiment with Architectures / 实验不同架构:**
   - Different numbers of layers
   - Different activation functions
   - Different learning rates

---

## Reference Resources / 参考资源

1. **Textbooks / 教科书:**
   - Deep Learning (Goodfellow et al.)
   - Neural Networks and Deep Learning (Nielsen)

2. **Online Resources / 在线资源:**
   - Stanford CS231n Course Notes
   - PyTorch/TensorFlow Tutorials

---

## 练习题 / Practice Problems

### 问题1 / Problem 1:
推导反向传播算法中输出层的梯度计算公式。
Derive the gradient calculation formula for output layer in backpropagation algorithm.

### 问题2 / Problem 2:
解释为什么ReLU激活函数比sigmoid函数在深度网络中更常用。
Explain why ReLU activation function is more commonly used than sigmoid in deep networks.

### 问题3 / Problem 3:
比较前向传播和反向传播的计算复杂度。
Compare the computational complexity of forward propagation and backpropagation.

---

## 例题与解答 / Worked Examples

### 例题1：前向传播计算 / Forward Propagation Calculation

**题目 / Question:**  
给定一个简单的神经网络，计算前向传播。
Given a simple neural network, calculate forward propagation.

网络结构 / Network structure:
- 输入层：2个神经元，输入x = [1, 2] / Input layer: 2 neurons, input x = [1, 2]
- 隐藏层：2个神经元，权重W = [[0.5, 0.3], [0.2, 0.4]]，偏置b = [0.1, 0.2] / Hidden layer: 2 neurons, weights W = [[0.5, 0.3], [0.2, 0.4]], bias b = [0.1, 0.2]
- 激活函数：sigmoid / Activation function: sigmoid

**详细解答 / Detailed Solution:**

**步骤1：计算隐藏层输入 / Step 1: Calculate Hidden Layer Input**
$$
\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b} = \begin{bmatrix} 0.5 & 0.3 \\ 0.2 & 0.4 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix}
$$

$$
= \begin{bmatrix} 0.5 \times 1 + 0.3 \times 2 \\ 0.2 \times 1 + 0.4 \times 2 \end{bmatrix} + \begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix} = \begin{bmatrix} 1.1 \\ 1.0 \end{bmatrix} + \begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix} = \begin{bmatrix} 1.2 \\ 1.2 \end{bmatrix}
$$

**步骤2：应用激活函数 / Step 2: Apply Activation Function**
$$
\mathbf{a} = \sigma(\mathbf{z}) = \begin{bmatrix} \sigma(1.2) \\ \sigma(1.2) \end{bmatrix} = \begin{bmatrix} \frac{1}{1+e^{-1.2}} \\ \frac{1}{1+e^{-1.2}} \end{bmatrix} \approx \begin{bmatrix} 0.769 \\ 0.769 \end{bmatrix}
$$

**结论 / Conclusion:**
隐藏层的激活值约为[0.769, 0.769]。
Hidden layer activations are approximately [0.769, 0.769].

---

### 例题2：反向传播梯度计算 / Backpropagation Gradient Calculation

**题目 / Question:**  
对于简单的两层网络，推导损失函数对权重的梯度。
For a simple two-layer network, derive the gradient of loss function with respect to weights.

网络：输入x → 隐藏层h → 输出y / Network: input x → hidden layer h → output y
损失函数：L = (y - t)²，其中t是目标值 / Loss function: L = (y - t)², where t is target

**详细解答 / Detailed Solution:**

**步骤1：前向传播 / Step 1: Forward Propagation**
$$
h = \sigma(w_1 x + b_1)
$$

$$
y = w_2 h + b_2
$$

$$
L = (y - t)^2
$$

**步骤2：计算输出层梯度 / Step 2: Calculate Output Layer Gradient**
$$
\frac{\partial L}{\partial y} = 2(y - t)
$$

$$
\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial w_2} = 2(y - t) \times h
$$

$$
\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial b_2} = 2(y - t) \times 1 = 2(y - t)
$$

**步骤3：计算隐藏层梯度 / Step 3: Calculate Hidden Layer Gradient**
$$
\frac{\partial L}{\partial h} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial h} = 2(y - t) \times w_2
$$

$$
\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial h} \frac{\partial h}{\partial w_1} = 2(y - t) w_2 \times \sigma'(w_1 x + b_1) \times x
$$

$$
\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial h} \frac{\partial h}{\partial b_1} = 2(y - t) w_2 \times \sigma'(w_1 x + b_1)
$$

**结论 / Conclusion:**
通过链式法则，可以从输出层向输入层逐层计算梯度。
Through chain rule, gradients can be calculated layer by layer from output layer to input layer.

---

