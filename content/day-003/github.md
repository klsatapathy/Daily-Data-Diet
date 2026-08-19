# Day 3 — Variables & Data Types in Python

**Category:** Python Pantry · **Difficulty:** Beginner · **Tags:** python-pantry, variables

---

## Definition
A variable is a named reference bound to a memory address containing a specific object. A data type is an attribute of an object that dictates what operations can be performed on it, how it is stored in memory, and how the interpreter evaluates it. Python uses dynamic typing, meaning variable bindings are resolved at runtime and variables themselves do not have explicit type declarations; rather, the underlying objects possess types.

## Rule of Thumb
Variables are pointers to objects, not storage containers, and data types define the capabilities of those referenced objects.

## Technical Comparison

| Data Category | Mutability | Ordered | Primary Use Case |
| :--- | :--- | :--- | :--- |
| `int` / `float` | Immutable | N/A | Numerical computation, counting, indexing |
| `str` | Immutable | Yes | Text representation, categorical identifiers |
| `list` | Mutable | Yes | Ordered sequences of heterogeneous items |
| `dict` | Mutable | No (Key-Order preserved in Python 3.7+) | Fast key-value lookups, JSON-like records |

## Code Example

```python
# Type checking and dynamic reassignment
user_count: int = 42
print(type(user_count))  # <class 'int'>

user_count = "forty-two"
print(type(user_count))  # <class 'str'>

# Collection operations and mutation
metrics = {"accuracy": 0.95, "epochs": 10}
metrics["accuracy"] = 0.97  # Dictionary mutation
```

## Best Practices
- Explicitly declare type hints for function signatures and critical variables to improve readability and static analysis.
- Prefer immutable data types (tuples, frozensets) when data integrity must be guaranteed across functions.
- Avoid using mutable objects (lists, dictionaries) as default arguments in function definitions.
- Leverage built-in functions such as `isinstance()` rather than `type()` when performing type checks.

## Common Interview Question
**Q:** What is the difference between mutability and immutability in Python, and how does it affect variable assignment?
**A:** Mutable objects (such as lists and dictionaries) can have their internal state modified in place without altering their memory address. Immutable objects (such as integers, floats, strings, and tuples) cannot be altered once created; any operation that modifies their value actually instantiates a new object in memory. When assigning variables, mutable objects allow multiple variables to reference the exact same underlying object (leading to side effects if modified), whereas reassignment of an immutable variable simply binds that variable name to a new memory address.

## Summary
Python variables act as dynamic references to typed objects stored in memory. Understanding the distinction between mutable and immutable types is critical for managing state, preventing unintended side effects, and optimizing data structures in analytical workflows.
