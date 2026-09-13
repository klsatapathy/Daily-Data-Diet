# Day 6 — What is Machine Learning, Really?

**Category:** ML Main Course · **Difficulty:** Beginner · **Tags:** ml-main-course, what

---

## Definition
Machine learning (ML) is a subfield of artificial intelligence focused on developing algorithms that estimate an unknown target function $f: X \to Y$ using empirical data. Instead of relying on explicitly programmed, deterministic rules, machine learning systems optimize internal parameters via numerical and statistical techniques to minimize a predefined objective or loss function.

## Rule of Thumb
If the rules defining the output can be completely specified by domain logic, write software; if the rules must be extracted statistically from data patterns, use machine learning.

## Technical Comparison

| Attribute | Traditional Software Engineering | Machine Learning |
| :--- | :--- | :--- |
| **Input** | Structured Data + Handcrafted Rules | Observed Data + Expected Targets (Labels) |
| **Output** | Deterministic Results | Statistical Model (Learned Parameters / Functions) |
| **Logic Construction** | Explicitly coded by human engineers | Statistically estimated via optimization algorithms |
| **Handling Edge Cases** | Requires manual logic updates and conditionals | Requires representative retraining data |
| **Core Assumption** | System behavior is entirely specifiable | Underlying distribution produces discoverable patterns |

## Code Example

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Features (X) and Targets (y)
X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([3.0, 5.0, 7.0, 9.0, 11.0])  # Underlying relation: y = 2x + 1

# Initialize estimator and learn parameters from data
model = LinearRegression()
model.fit(X, y)

# Inspect optimized parameters
weight = model.coef_[0]
bias = model.intercept_

# Perform inference on unseen input
X_new = np.array([[6.0]])
prediction = model.predict(X_new)

print(f"Learned Function: y = {weight:.2f}x + {bias:.2f}")
print(f"Inference for X=6.0: {prediction[0]:.2f}")
```

## Best Practices
* **Construct a Simple Baseline First:** Evaluate performance against a trivial rule-based heuristic, running average, or dummy classifier before training complex models.
* **Isolate Data Splits Early:** Divide datasets into distinct training, validation, and test partitions before any data transformations to prevent data leakage.
* **Verify Distributional Assumptions:** Confirm that the underlying data generating process is stationary; monitor continuously for covariate shift and concept drift in production.
* **Align Loss Functions with Objectives:** Ensure the numerical loss being minimized (e.g., Cross-Entropy, Mean Squared Error) directly maps to the operational evaluation metric.

## Common Interview Question
**
