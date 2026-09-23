# Day 6 — What is Machine Learning, Really?

**Category:** ML Main Course · **Difficulty:** Beginner · **Tags:** ml-main-course, what

---

## Definition

Machine Learning (ML) is a subfield of artificial intelligence focused on developing mathematical models and statistical algorithms that optimize a performance metric by identifying patterns and functional relationships within empirical data ($f(X) \approx y$), rather than executing explicitly encoded deterministic rules.

## Rule of Thumb

If the decision logic can be fully specified using deterministic `if-else` rules without excessive complexity or edge-case degradation, do not use machine learning.

## Technical Comparison

| Dimension | Traditional Programming | Machine Learning |
| :--- | :--- | :--- |
| **Primary Inputs** | Data + Explicit Rules/Algorithms | Data + Observed Outputs (or Objectives) |
| **Primary Output** | Computed Results | Inferred Logic / Mathematical Model ($f(X)$) |
| **Logic Formulation** | Handcrafted by domain experts/engineers | Learned via optimization of an objective function |
| **Adaptability** | Requires manual code updates for new patterns | Retrains or updates parameters on new data distributions |
| **Failure Mode** | Logic bugs, unhandled exceptions, rigid failures | Distribution shift, poor generalization, silent degradation |

## Code Example

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Feature matrix (X) and continuous target vector (y)
X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([2.1, 3.9, 6.2, 8.1, 9.9])

# Instantiate and optimize model parameters (find weight w and bias b for y = wX + b)
model = LinearRegression()
model.fit(X, y)

# Evaluate learned mapping on unseen inputs
X_new = np.array([[6.0], [7.0]])
predictions = model.predict(X_new)

print(f"Learned Coefficient (Weight): {model.coef_[0]:.2f}")
print(f"Learned Intercept (Bias): {model.intercept_:.2f}")
print(f"Predictions for {X_new.flatten()}: {predictions.round(2)}")
```

## Best Practices

* **Establish a baseline first:** Benchmark ML implementations against a deterministic heuristic, historical average, or linear model before deploying complex architectures.
* **Isolate evaluation data:** Partition datasets into distinct train, validation, and test splits prior to any preprocessing to prevent data leakage.
* **Validate data distributions:** Ensure the distribution of features ($P(X)$) and targets ($P(y|X)$) in the training dataset closely mirrors the production environment.
* **Prioritize data quality over model complexity:** Address missing values, label noise, and feature representation before tuning hyperparameters.

## Common Interview Question

**Q:** What is the fundamental difference between model parameters and model hyperparameters?  
**A:** Model parameters are internal variables whose values are estimated or learned directly from training data through an optimization algorithm (e.g., regression coefficients, neural network weights). Model hyperparameters are external configuration variables that must be set manually or via cross-validation before training begins, as they dictate how the training algorithm executes (e.g., learning rate, regularization coefficient, maximum tree depth).

## Summary

Machine learning provides a computational framework for approximating unknown functions directly from empirical observations through parameter optimization. It replaces manual rule creation with automated pattern discovery, enabling systems to generalize to unobserved data instances. It is suitable when functional dependencies are too nonlinear, high-dimensional, or volatile to encode deterministically.
