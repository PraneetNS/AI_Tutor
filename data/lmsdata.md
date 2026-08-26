# LMS Course Data Fixtures

## Course 101: Machine Learning
- **Course ID**: 101
- **Course Title**: Machine Learning
- **Description**: Introduction to Machine Learning concepts, algorithms, and practical applications.

### Lesson 10: ML Basics
- **Lesson ID**: 10
- **Lesson Name**: ML Basics

#### Lecture 50: Supervised Learning
- **Lecture ID**: 50
- **Type**: video
- **Content Excerpts / Chunks**:
  - **Chunk 50-1**: Supervised learning is a machine learning paradigm where models are trained on labeled data pairs $(x, y)$. The objective is to learn a mapping function $f(x) \approx y$ that generalizes to unseen test examples. Examples include linear regression for continuous targets and logistic regression or decision trees for discrete classification.
  - **Chunk 50-2**: In supervised learning, the model makes predictions on training instances and evaluates errors using a loss function like Mean Squared Error (MSE) for regression or Cross-Entropy for classification. Optimization algorithms adjust weights to minimize this loss.

#### Lecture 51: Unsupervised Learning
- **Lecture ID**: 51
- **Type**: video
- **Content Excerpts / Chunks**:
  - **Chunk 51-1**: Unsupervised learning operates on unlabeled data $\{x_i\}$. The algorithm discovers hidden structure, clusters, or lower-dimensional representations without explicit target outputs. Key algorithms include K-Means clustering, Hierarchical clustering, and Principal Component Analysis (PCA).

### Lesson 20: Optimization & Model Training
- **Lesson ID**: 20
- **Lesson Name**: Optimization & Model Training

#### Lecture 60: Gradient Descent & Cost Functions
- **Lecture ID**: 60
- **Type**: video
- **Content Excerpts / Chunks**:
  - **Chunk 60-1**: Gradient descent is an iterative first-order optimization algorithm used to minimize a differentiable cost function $J(\theta)$. The update rule is $\theta \leftarrow \theta - \alpha \nabla J(\theta)$, where $\alpha$ is the learning rate. If $\alpha$ is too high, gradient descent may oscillate or diverge; if too low, convergence is excessively slow.
  - **Chunk 60-2**: Batch gradient descent calculates gradients over the entire dataset, Stochastic Gradient Descent (SGD) uses one sample per step, and Mini-batch SGD strikes a balance by computing gradients over small batches (e.g., 32 or 64 samples).
