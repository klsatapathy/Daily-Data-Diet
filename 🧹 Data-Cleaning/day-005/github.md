# Day 5 — Why Dirty Data Kills Models

**Category:** Data Cleaning · **Difficulty:** Beginner · **Tags:** data-cleaning, why

---

## Definition
Dirty data refers to datasets containing anomalies, errors, missing values, duplicates, inconsistencies, or formatting errors that deviate from expected data types and schemas, impairing the statistical validity and predictive performance of machine learning models.

## Rule of Thumb
Model performance is bound by data quality; algorithmic complexity cannot compensate for corrupted training inputs.

## Technical Comparison

| Issue Type | Symptom | Impact on Model |
| :--- | :--- | :--- |
| **Missing Values** | Null or NaN entries in feature matrices | Training failure or biased imputation |
| **Outliers / Noise** | Extreme values beyond expected distributions | Distorted gradient updates and skewed coefficients |
| **Type Inconsistencies** | Mixed types (e.g., strings in numeric columns) | Runtime exceptions or incorrect parsing |
| **Duplicate Rows** | Exact or near-exact redundant records | Data leakage between train and validation splits |

## Code Example

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Clean data generation
np.random.seed(42)
X_clean = np.linspace(0, 10, 100).reshape(-1, 1)
y_clean = 2.5 * X_clean.squeeze() + 1.0 + np.random.normal(0, 0.5, 100)

# Introduction of dirty data: missing values, string contamination, and extreme outliers
X_dirty = X_clean.copy()
X_dirty[10:15] = np.nan
X_dirty[20] = "invalid_string"

y_dirty = y_clean.copy()
y_dirty[5] = 1000.0  # Extreme outlier

df = pd.DataFrame({"feature": X_dirty.flatten(), "target": y_dirty})

# Attempting to fit a model on dirty data results in a TypeError
try:
    df["feature"] = pd.to_numeric(df["feature"], errors="coerce")
    df = df.dropna()
    
    X = df[["feature"]]
    y = df["target"]
    
    model = LinearRegression()
    model.fit(X, y)
    print("Model fitted successfully after cleaning.")
except Exception as e:
    print(f"Training failed due to dirty data: {e}")
```

## Best Practices
- Enforce strict schema validation at data ingestion points using tools like Pydantic or Great Expectations.
- Implement automated imputation and anomaly detection pipelines before exploratory data analysis.
- Split data into training and validation sets prior to performing any missing value imputation or scaling to prevent data leakage.
- Maintain version control for datasets to track cleaning steps and ensure reproducibility.

## Common Interview Question
**Q:** How do missing values impact tree-based models differently than linear models, and how should you handle them for each?
**A:** Linear models cannot handle missing values natively and require imputation (such as mean, median, or model-based prediction) or complete-case deletion to prevent runtime errors and coefficient bias. Tree-based models, such as XGBoost or LightGBM, can handle missing values natively by evaluating both directions of a split for missing entries to find the optimal gain. For linear models, use explicit imputation strategies fitted strictly on training folds; for tree-based models, leaving values as null or passing them through native handlers is often optimal.

## Summary
Dirty data introduces systemic bias, distorts parameter estimation, and causes runtime failures in machine learning pipelines. Rigorous data cleaning, schema validation, and disciplined preprocessing protocols are prerequisites for reliable model training and inference.
