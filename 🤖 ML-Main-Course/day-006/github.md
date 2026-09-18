# Day 6 — What is Machine Learning, Really?

**Category:** ML Main Course · **Difficulty:** Beginner · **Tags:** ml-main-course, what

---

## Definition
Machine learning is a subset of artificial intelligence focused on algorithms that parse data, identify statistical patterns within that data, and optimize their own parameters to make predictions or decisions on unseen data without being explicitly programmed for every specific scenario.

## Rule of Thumb
If a system requires hardcoded conditional logic to handle new inputs, it is traditional software; if it adapts its internal weights using historical data to improve performance on a metric, it is machine learning.

## Technical Comparison

| Dimension | Traditional Programming | Machine Learning |
| :--- | :--- | :--- |
| **Input** | Data + Explicit Rules | Data + Desired Outputs (Labels) |
| **Execution** | Rules + Data $\rightarrow$ Output | Data + Outputs $\rightarrow$ Inferred Rules |
| **Maintenance** | Manual rule updates for edge cases | Retraining on new data distributions |
| **Determinism** | Deterministic outputs for identical code/inputs | Probabilistic outputs governed by parameter weights |

## Code Example

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Feature matrix X (e.g., square footage) and target vector y (e.g., price)
X = np.array([[650], [800], [1200], [1500]])
y = np.array([180000, 220000, 310000, 380000])

# Initialize and fit the machine learning model
model = LinearRegression()
model.fit(X, y)

# Predict target for unseen input data
new_house = np.array([[1100]])
predicted_price = model.predict(new_house)

print(f"Predicted Price: ${predicted_price[0]:,.2f}")
```

## Best Practices
- Separate datasets strictly into training, validation, and test sets prior to feature engineering to prevent data leakage.
- Establish a robust baseline model using simple heuristics or linear methods before deploying complex architectures.
- Track model metrics, hyperparameters, and dataset versions systematically using experiment tracking tools.
- Monitor deployed models continuously for concept drift and data drift to ensure sustained prediction accuracy.

## Common Interview Question
**Q:** What is the fundamental difference between parametric and non-parametric machine learning algorithms?
**A:** Parametric models make strong assumptions about the underlying data distribution and summarize data with a fixed number of parameters regardless of the training set size (e.g., Logistic Regression). Non-parametric models make fewer assumptions about the functional form and allow the number of parameters to grow with the amount of training data, often storing or utilizing the training instances directly during prediction (e.g., K-Nearest Neighbors).

## Summary
Machine learning automates the discovery of mathematical relationships within datasets to generalize predictions to new observations. By shifting from explicit instruction-following to statistical optimization, systems can scale to complex pattern-recognition tasks. Success in machine learning depends heavily on rigorous data preparation, careful evaluation methodology, and ongoing model monitoring.
