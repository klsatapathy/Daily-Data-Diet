<!-- Auto-drafted starter note (original GitHub write-up for Day 1 wasn't supplied) — review and edit before treating as final. -->

# Day 1 — Data Analytics vs Data Science

**Category:** Fundamentals · **Difficulty:** Beginner · **Tags:** data-science, data-analytics, python, career

---

## Definition

Data Analytics is the practice of examining historical data to describe and explain what has already happened. Data Science extends this by using statistics, machine learning, and programming to build models that predict or influence what will happen next.

## Rule of Thumb

If the question starts with "What happened?" or "Why did it happen?", it's Analytics. If it starts with "What will happen?" or "What should we do about it?", it's Data Science.

## Technical Comparison

| Aspect | Data Analytics | Data Science |
|---|---|---|
| Time orientation | Past (descriptive) | Future (predictive/prescriptive) |
| Core question | What happened? | What will happen? |
| Typical tools | Excel, SQL, Tableau, Power BI | Python, R, ML libraries, statistics |
| Output | Reports, dashboards, summaries | Models, predictions, automated decisions |
| Skill emphasis | Querying, visualization, business context | Programming, statistics, model building |

## Code Example

```python
import pandas as pd

# Analytics: describe what already happened
sales = pd.read_csv("sales.csv")
last_month_total = sales[sales["month"] == "2025-12"]["revenue"].sum()
print(f"Last month's revenue: {last_month_total}")

# Data Science: predict what happens next
from sklearn.linear_model import LinearRegression

X = sales[["month_number"]]
y = sales["revenue"]
model = LinearRegression().fit(X, y)
next_month_prediction = model.predict([[X["month_number"].max() + 1]])
print(f"Predicted next month's revenue: {next_month_prediction[0]}")
```

## Best Practices

- Don't treat these as competing fields — most real data roles blend both.
- Start with Analytics (understanding the data) before jumping into Science (modeling it) — a model built on data you don't understand will mislead you.
- Learn Python early even if your first job is analytics-heavy; it's the bridge between the two.

## Common Interview Question

**Q:** How would you explain the difference between a Data Analyst and a Data Scientist to a non-technical stakeholder?
**A:** A Data Analyst tells you what already happened using past data (like a rearview mirror), while a Data Scientist builds models that forecast what's likely to happen next (like a GPS or weather forecast) — both are essential, but they answer different questions.

## Summary

Data Analytics and Data Science aren't rivals — they're two ends of the same pipeline. Analytics explains the past so decisions can be grounded in reality; Data Science uses that same data to look forward. Python is the common thread that connects both.
