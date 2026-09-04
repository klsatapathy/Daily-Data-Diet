# Day 6 — What is Machine Learning, Really?

**Category:** ML Main Course · **Difficulty:** Beginner · **Tags:** ml-main-course, what

---

## Definition
Machine learning is a subset of artificial intelligence focused on algorithms that parse data, learn from that data, and apply what they have learned to make informed decisions or predictions without being explicitly programmed for every specific outcome. Mathematically, it involves optimizing parameters within a defined objective function by minimizing a loss function over a given dataset.

## Rule of Thumb
Traditional programming uses data and rules to output answers; machine learning uses data and answers to output rules.

## Technical Comparison

| Dimension | Supervised Learning | Unsupervised Learning | Reinforcement Learning |
| :--- | :--- | :--- | :--- |
| **Data Requirements** | Feature matrix $X$ and ground-truth labels $y$ | Feature matrix $X$ only (no labels) | States, actions, and reward signals |
| **Primary Objective** | Regression or classification | Clustering, dimensionality reduction, density estimation | Policy optimization to maximize cumulative reward |
| **Evaluation Metrics** | MSE, RMSE, Accuracy, Precision, Recall, F1 | Silhouette score, Inertia, Reconstruction error | Cumulative reward, Value function convergence |
| **Common Algorithms** | Linear Regression, Random Forest, XGBoost | K-Means, PCA, DBSCAN | Q-Learning, PPO, DQN |

## Code Example

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic input data (X) and target outputs (y)
X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([2.0, 4.0, 6.0, 8.0])

# Initialize the model instance
model = LinearRegression()

# Fit model parameters (weights and bias) to the training data
model.fit(X, y)

# Generate predictions on unseen input data
y_pred = model.predict(np.array([[5.0]]))

# Evaluate model error
mse = mean_squared_error(y, model.predict(X))

print(f"Prediction for X=5.0: {y_pred[0]:.2f}")
print(f"Training Mean Squared Error: {mse:.4f}")
```

## Best Practices
- Always partition data into training, validation, and test sets to prevent data leakage and ensure unbiased generalization metrics.
- Establish a naive baseline model (such as mean prediction or zero-rule classification) before deploying complex algorithms.
- Scale and normalize continuous numerical features to stabilize gradient descent convergence and prevent distance-based bias.
- Monitor models post-deployment for data drift and concept drift to maintain predictive accuracy over time.

## Common Interview Question
**Q:** What is the difference between inductive bias in machine learning and overfitting?
**A:** Inductive bias refers to the set of assumptions a learning algorithm uses to predict outputs for unseen inputs that it has not encountered during training (for example, the linearity assumption in linear regression). Overfitting occurs when a model learns the training data too well—including its underlying noise and outliers—resulting in high variance and poor generalization to unseen data, typically caused by excessive model complexity relative to the size of the training set.

## Summary
Machine learning automates the construction of analytical models by using optimization algorithms to discover patterns in data. Proper evaluation, rigorous validation protocols, and appropriate feature engineering are required to ensure these models generalize accurately to production environments.
