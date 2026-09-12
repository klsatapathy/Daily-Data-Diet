# Day 6 — What is Machine Learning, Really?

**Category:** ML Main Course · **Difficulty:** Beginner · **Tags:** ml-main-course, what

---

## Definition
Machine Learning (ML) is a subfield of artificial intelligence that focuses on constructing mathematical models whose parameters are iteratively optimized using empirical data. Rather than relying on static, hand-engineered decision logic, ML algorithms use statistical methods and optimization techniques (e.g., gradient descent) to infer functional mappings from input features to output targets.

## Rule of Thumb
If the operational logic can be fully defined by static deterministic rules, write code; if the logic must be derived from statistical variance in data, train a model.

## Technical Comparison

| Dimension | Traditional Programming | Machine Learning |
| :--- | :--- | :--- |
| **Primary Inputs** | Data + Explicit Rules | Data + Observed Outputs (Supervised) |
| **Primary Output** | Program Results / Predictions | Optimized Model (Rules / Parameters) |
| **Execution Model** | Deterministic control flow | Probabilistic inference |
| **Maintenance Cost** | Rule-base growth increases cyclomatic complexity | Data drift requires continuous retraining and validation |
| **Failure Modes** | Syntax errors, unhandled logic branches | Overfitting, data leakage, distribution shift |

## Code Example

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Feature matrix (X) and target vector (y)
X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([2.5, 4.5, 6.5, 8.5])

# Initialize and optimize parameters (weight and bias) via Ordinary Least Squares
model = LinearRegression()
model.fit(X, y)

# Infer output for unseen input
X_new = np.array([[5.0]])
prediction = model.predict(X_new)

print(f"Learned Weight: {model.coef_[0]:.2f}")
print(f"Learned Bias: {model.intercept_:.2f}")
print(f"Prediction for X=5.0: {prediction[0]:.2f}")
```

## Best Practices
* **Establish a Non-ML Baseline:** Always benchmark models against simple heuristics or standard statistical aggregates (e.g., median, mode, or simple conditional rules) before adopting complex algorithms.
* **Prevent Data Leakage:** Ensure data preprocessing, imputation, and scaling parameters are calculated solely on training splits before transforming validation or test sets.
* **Select Metrics Aligned with Target Distribution:** Avoid using raw accuracy on imbalanced datasets; use precision, recall, PR-AUC, or cost-weighted loss functions depending on the operational impact of false positives versus false negatives.
* **Track Data and Model Drift:** Implement production monitoring to detect statistical divergence between the training data distribution and incoming inference data.

## Common Interview Question
**Q:** What is the difference between an algorithm and a model in machine learning?  
**A:** An algorithm is a sequence of mathematical operations and optimization rules used to learn patterns from data (e.g., Stochastic Gradient Descent, Random Forest algorithm). A model is the specific computational artifact produced after running an algorithm on a concrete dataset; it contains the finalized architecture, learned parameters (e.g., weights and biases), and logic required to generate predictions on new inputs.

## Summary
Machine learning replaces manual rule construction with algorithmic parameter estimation derived from data distributions. It provides scalable solutions for complex, high-dimensional problems where deterministic programming is infeasible. Proper implementation requires rigorous dataset splitting, metric alignment, and continuous validation to guard against non-deterministic failure modes.
