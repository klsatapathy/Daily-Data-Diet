# Day 6 — What is Machine Learning, Really?

**Category:** ML Main Course · **Difficulty:** Beginner · **Tags:** ml-main-course, what

---

## Definition

Machine Learning (ML) is a branch of artificial intelligence that constructs mathematical models from empirical data to solve tasks such as prediction, classification, and pattern recognition. Formally, an algorithm is said to learn from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$. Instead of executing explicitly programmed deterministic instructions, ML optimizes the parameters of an approximating function $f(X) \approx Y$ using numerical optimization and statistical inference.

## Rule of Thumb

If the decision boundaries can be fully resolved with deterministic conditional logic (`if/else`), use classical programming; if the underlying mapping function must be approximated from statistical distributions, use machine learning.

## Technical Comparison

| Dimension | Classical Software Engineering | Machine Learning |
| :--- | :--- | :--- |
| **Primary Inputs** | Data + Explicit Logic (Code) | Data + Observed Outcomes (Labels/Targets) |
| **Primary Output** | Program Results / Output State | Trained Model Parameters (Approximation Function) |
| **Problem Complexity** | Best for deterministic, well-bounded logic | Best for probabilistic, high-dimensional, or perceptual tasks |
| **Handling Edge Cases** | Requires manual branching and code refactoring | Requires representative retraining data or regular
