# Day 4 — Mean, Median, Mode - Which One Lies?

**Category:** Stats Snacks · **Difficulty:** Beginner · **Tags:** stats-snacks, mean

---

## Definition
The mean is the arithmetic average of a dataset, calculated as the sum of all values divided by the total count ($n$). The median is the middle value in an ordered dataset; for an even $n$, it is the arithmetic mean of the two central values. The mode is the value that appears most frequently in the dataset.

## Rule of Thumb
Use the mean for symmetric distributions, the median for skewed distributions, and the mode for categorical distributions.

## Technical Comparison

| Metric | Sensitivity to Outliers | Mathematical Definition | Computational Complexity | Suitable Data Types |
| :--- | :--- | :--- | :--- | :--- |
| **Mean** | High (Affected by extreme values) | $\frac{1}{n} \sum_{i=1}^{n} x_i$ | $O(n)$ | Continuous, Discrete (Numeric) |
| **Median** | Low (Robust to extreme values) | Middle value of sorted array | $O(n \log n)$ due to sorting | Continuous, Discrete, Ordinal |
| **Mode** | None | $\arg\max_x f(x)$ | $O(n)$ | Nominal, Ordinal, Discrete |

## Code Example

```python
import numpy as np
from scipy import stats

# Dataset with an extreme outlier
data = np.array([10, 12, 14, 15, 15, 18, 20, 1000])

mean_val = np.mean(data)
median_val = np.median(data)
mode_val = stats.mode(data, keepdims=True).mode[0]

print(f"Mean: {mean_val}")
print(f"Median: {median_val}")
print(f"Mode: {mode_val}")
```

## Best Practices
- Inspect the distribution skewness using histograms or density plots before selecting a central tendency metric.
- Report the median and Interquartile Range (IQR) alongside the mean and standard deviation for skewed numerical distributions.
- Handle missing values explicitly prior to calculating central tendency metrics, as `NaN` values propagate through mean calculations.
- Recognize that multimodal distributions render the mode ambiguous; consider kernel density estimation for identifying multiple local maxima.

## Common Interview Question
**Q:** When would you deliberately choose to report the median instead of the mean, and what information is lost by doing so?
**A:** The median is reported instead of the mean when the dataset is heavily skewed or contains extreme outliers that would pull the mean away from the center of the majority of the data (e.g., household income or real estate prices). The information lost by using the median is the exact magnitude of the extreme values, as the median only accounts for the relative ranking of the data points rather than their absolute numerical scale.

## Summary
The mean, median, and mode measure central tendency differently and behave predictably based on data distribution and skewness. Selecting the appropriate metric prevents skewed data from distorting analytical conclusions.
