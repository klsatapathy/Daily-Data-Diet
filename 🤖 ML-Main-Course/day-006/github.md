# Day 6 — What is Machine Learning, Really?

**Category:** ML Main Course · **Difficulty:** Beginner · **Tags:** ml-main-course, what

---

## Definition

Machine Learning (ML) is a subfield of artificial intelligence that focuses on mathematical algorithms capable of identifying statistical patterns and functional relationships directly from empirical data. Formally defined by Tom Mitchell (1997), a computer program is said to learn from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$. Rather than relying on hardcoded procedural instructions, ML algorithms optimize internal parameters via numerical methods to minimize error or maximize likelihood on observed inputs.

## Rule of Thumb

Use traditional programming when deterministic rules fully describe the system; use machine learning when the system's logic is too complex to specify manually, but sufficient historical data exists to infer it.

## Technical Comparison

| Dimension | Traditional Programming | Machine Learning |
| :--- | :--- | :--- |
| **Primary Inputs** | Data + Explicit Rules (Code) | Data + Expected Outcomes (Labels) |
| **Output** | Answers / State Changes | Predictive Model (Function approximation / Weights) |
| **Logic Construction** | Manually engineered by human programmers | Algorithmically discovered via numerical optimization |
| **Adaptability** | Requires manual code refactoring for edge cases | Adapts via retraining on updated data distributions |
| **Explainability** | High; deterministic execution pathways | Variable; often probabilistic and high-dimensional |
| **Error Modes** | Syntax errors, logic bugs, unhandled exceptions | Bias, variance, concept drift, data leakage |

## Code Example

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Empirical data: Features (X) and Target (y)
# Relationship: y = 2x + 1 + noise
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3.1, 4.9, 7.2, 8.8, 11.1])

# Initialize the learning algorithm
model = LinearRegression()

# Training step: Optimize internal parameters (weights and bias) using data
model.fit(X, y)

# Learned parameters derived from experience
weight = model.coef_[0]
bias = model.intercept_

# Inference step: Evaluate new inputs using learned parameters
new_data = np.array([[6]])
prediction = model.predict(new_data)

print(f"Learned Weight: {weight:.2f}")
print(f"Learned Bias:   {bias:.2f}")
print(f"Prediction for X=6: {prediction[0]:.2f}")
```

## Best Practices

* **Establish a Heuristic Baseline:** Always benchmark ML models against a simple, deterministic rule or standard statistical summary (e.g., mean, median, hardcoded thresholds) to justify complexity.
* **Isolate Datasets Prior to Processing:** Strictly separate training, validation, and test sets before any feature engineering or scaling to prevent data leakage.
* **Align Loss Functions with Domain Objectives:** Choose loss functions that accurately penalize real-world error costs (e.g., Mean Absolute Error for outlier robustness vs. Mean Squared Error for penalizing large deviations).
* **Monitor Post-Deployment Distributions:** Monitor inference inputs for covariate and concept drift, as ML model performance degrades when production distributions deviate from training data.

## Common Interview Question

**Q:** What is the fundamental difference between traditional software engineering and machine learning, and when should you avoid using machine learning?

**A:** Traditional software engineering takes input data and hand-coded rules to produce outputs, making execution paths strictly deterministic and debuggable. Machine learning takes inputs and target outputs to learn an approximating function (parameters), making execution probabilistic. 

Machine learning should be avoided when:
1. The problem can be fully solved using deterministic logic, regular expressions, or simple lookup tables.
2. The tolerance for error is zero and outputs must be 100% auditable and reproducible without probabilistic variance.
3. Insufficient, highly biased, or unrepresentative data exists to train an objective model.
4. The compute, data infrastructure, and monitoring costs outweigh the marginal performance gains over a baseline heuristic.

## Summary

Machine Learning replaces manually engineered program logic with statistical models whose parameters are iteratively tuned using historical data and mathematical optimization. It is optimal for high-dimensional, complex patterns where explicit computational instructions are intractable to write by hand. Deploying ML introduces technical trade-offs, shifting engineering complexity from writing code logic to managing data pipelines, monitoring distribution shifts, and evaluating probabilistic performance.
