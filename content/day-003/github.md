# Day 3 — Variables & Data Types in Python

**Category:** Python Pantry · **Difficulty:** Beginner · **Tags:** python-pantry, variables

---

## Definition
A variable in Python is an identifier that references an object in memory. Python is dynamically and strongly typed: variable types are inferred and bound to runtime objects rather than explicitly declared, and implicit type coercion is restricted to prevent invalid operations between incompatible types.

## Rule of Thumb
Variables are labeled pointers to memory objects, not typed storage boxes.

## Technical Comparison

| Data Type | Category | Mutability | Definition / Purpose | Example |
| :--- | :--- | :--- | :--- | :--- |
| `int` | Numeric | Immutable | Arbitrary-precision integers | `x = 42` |
| `float` | Numeric | Immutable | Double-precision floating-point numbers | `x = 3.14159` |
| `str` | Sequence | Immutable | Unicode text character sequences | `x = "data"` |
| `bool` | Boolean | Immutable | Truth values representing `True` or `False` | `x = True` |
| `list` | Sequence | Mutable | Ordered, heterogeneous collection of objects | `x = [1, "two", 3.0]` |
| `tuple` | Sequence | Immutable | Ordered, heterogeneous fixed collection | `x = (1, 2, 3)` |
| `dict` | Mapping | Mutable | Unordered/insertion-ordered key-value pairs | `x = {"a": 1, "b": 2}` |
| `set` | Set | Mutable | Unordered collection of unique, hashable objects | `x = {1, 2, 3}` |

## Code Example

```python
# Variable assignment and dynamic reassignment
record_id: int = 101
feature_value: float = 23.85
is_active: bool = True
label: str = "cluster_1"

# Type inspection
print(type(record_id))  # Output: <class 'int'>
print(isinstance(feature_value, float))  # Output: True

# Dynamic typing
record_id = "ID_101"
print(type(record_id))  # Output: <class 'str'>

# Type casting
raw_count = "450"
processed_count = int(raw_count)

# Mutability impact
list_a = [1, 2, 3]
list_b = list_a
list_b.append(4)
print(list_a)  # Output: [1, 2, 3, 4] (same object reference)
```

## Best Practices
* Use descriptive names in `snake_case` according to PEP 8 standards (e.g., `raw_data_path`, `user_id_list`).
* Implement Python type hints (`typing` module) to improve code readability, validation, and static analysis.
* Avoid using Python built-in function and type names as variable identifiers (e.g., `list`, `dict`, `str`, `id`, `type`).
* Ensure immutability for fixed reference data by using `tuple` or `frozenset` instead of `list` or `set` to prevent accidental state modification.

## Common Interview Question
**Q:** How does Python handle variable assignment and memory management for mutable versus immutable types when passed into a function?  
**A:** Python uses "pass-by-object-reference" (or "call-by-assignment"). When a variable is passed to a function, the function receives a copy of the reference to the underlying object. If the object is mutable (e.g., a `list` or `dict`), in-place modifications made inside the function directly alter the original object in memory. If the object is immutable (e.g., an `int` or `tuple`), any modification creates a new object and rebinds the local variable reference, leaving the original caller's object unchanged.

## Summary
Python variables act as dynamic references to memory addresses where data objects of specific types reside. Understanding the distinction between mutable and immutable data types is critical for managing data integrity and avoiding unintended side effects in data processing workflows. Mastering explicit casting, dynamic typing behavior, and reference mechanisms provides the baseline for writing reliable data engineering and analysis pipelines.
