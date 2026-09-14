# Day 6 — What is Machine Learning, Really?

**Category:** ML Main Course · **Difficulty:** Beginner · **Tags:** ml-main-course, what

---

## Definition
Machine learning is a subset of artificial intelligence focused on algorithms that parse data, identify statistical patterns within that data, and optimize their own parameters to make predictions or decisions on unseen data without being explicitly programmed for every specific outcome.

## Rule of Thumb
Machine learning automates the discovery of mapping functions from input features to target labels using historical data.

## Technical Comparison

| Dimension | Traditional Programming | Machine Learning |
| :--- | :--- | :--- |
| **Inputs** | Rules + Data | Data + Answers (Labels) |
| **Output** | Answers | Rules (Model Parameters) |
| **Maintenance** | Manual rule updates required for new edge cases | Automated parameter updates via retraining on new data |
| **Determinism** | Deterministic execution paths | Probabilistic outputs based on statistical inference |

## Code Example

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate synthetic data for a linear mapping function: y = 3x + 5
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([8, 11, 14, 17, 20])

# Initialize and train the machine learning model
model = LinearRegression()
model.fit(X, y)

# Predict outcomes for unseen data
X_new = np.array([[6], [7]])
predictions = model.predict(X_new)

print("Learned Coefficient:", model.coef_[0])
print("Learned Intercept:", model.intercept_)
print("Predictions:", predictions)
```

## Best Practices
- Separate datasets strictly into training, validation, and test sets to prevent data leakage and overoptimistic evaluation.
- Establish a naive baseline model (such as mean or median prediction) before deploying complex algorithms.
- Preprocess and normalize feature distributions to stabilize gradient descent convergence and distance calculations.
- Monitor deployed models continuously for concept drift and data drift to detect performance degradation over time.

## Common Interview Question
**Q:** What is the fundamental difference between supervised and unsupervised learning?
**A:** Supervised learning algorithms are trained using labeled datasets where both input features and the corresponding target outputs are provided to learn the mapping function. Unsupervised learning algorithms operate on unlabeled data, attempting to discover inherent structural properties, clusters, or lower-dimensional representations without explicit target guidance.

## Summary
Machine learning replaces manual rule creation with algorithmic optimization, utilizing historical data to estimate parameters for predictive inference. Proper evaluation, data partitioning, and baseline comparisons are critical for validating model generalization to unseen inputs.
