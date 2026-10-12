# Day 6 — What is Machine Learning, Really?

**Category:** ML Main Course · **Difficulty:** Beginner · **Tags:** ml-main-course, what

---

## Definition

Machine Learning (ML) is a branch of computational statistics and computer science wherein algorithms optimize mathematical parameters by optimizing an objective function over empirical data, rather than executing explicitly defined deterministic rules. Formally, a program is said to learn from experience $E$ with respect to a class of tasks $T$ and a performance metric $P$, if its measured performance at tasks in $T$ improves with experience $E$.

## Rule of Thumb

Use deterministic programming when the business logic consists of known, finite rules; use machine learning when the underlying mapping function must be approximated from empirical data.

## Technical Comparison

| Dimension | Classical Software Engineering | Machine Learning |
| :--- | :--- | :--- |
| **Primary Inputs** | Data + Explicit Rules (Code) | Data + Expected Outputs (Labels / Feedback) |
| **Primary Output** | Computed Result | Learned Function / Model Parameters |
| **Problem Formulation** | Deductive: applying known principles to data | Inductive: inferring general principles from specific cases |
| **Edge Case Handling** | Requires manual logic branches (`if/else`) | Requires representative data distribution |
| **Maintenance Burden** | Code maintenance as complexity increases | Data pipeline, feature drift, and retraining maintenance |

## Code Example

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Empirical data: Features (X) and Targets (y)
X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([3.0, 5.0, 7.0, 9.0])  # Underlying relationship: y = 2x + 1

# Initialize model and optimize parameters via empirical data
model = LinearRegression()
model.fit(X, y)

# Inspect learned parameters (approximated mapping function)
slope = model.coef_[0]
intercept = model.intercept_

# Generalize to unseen data
new_X = np.array([[5.0]])
prediction = model.predict(new_X)

print(f"Learned Equation: y = {slope:.1f}x + {intercept:.1f}")
print(f"Prediction for X=5.0: {prediction[0]:.1f}")
```

## Best Practices

*   **Establish a baseline:** Build a non-ML baseline (e.g., rule-based heuristic, median/mode predictor) before implementing complex models to quantify actual value-add.
*   **Enforce strict data splitting:** Separate data into training, validation, and test partitions before any feature engineering or parameter tuning to prevent data leakage.
*   **Align metrics with objectives:** Select loss functions and evaluation metrics (e.g., ROC-AUC, F1-score, RMSE) that directly reflect task-specific costs of false positives versus false negatives.
*   **Verify data representativeness:** Ensure training distributions mirror the expected inference environment to prevent model degradation due to distribution shift.

## Common Interview Question

**Q:** What is the difference between a heuristic (rule-based) system and a machine learning model, and when should you select one over the other?  
**A:** A heuristic system executes hardcoded, human-defined decision paths. It is ideal when domain rules are deterministic, edge cases are limited, and full interpretability is required. A machine learning model learns a mathematical mapping from features to outputs by optimizing an objective function over data. ML is preferred when the problem involves high-dimensional inputs, complex non-linear interactions, or evolving data patterns that make manual rule specification intractable.

## Summary

Machine learning reframes software development from explicit rule authoring to continuous parameter optimization over training data. It approximates an underlying data-generating distribution to enable generalization to unobserved inputs. The methodology is required when deterministic programming cannot practically scale to the complexity or variability of the task.
