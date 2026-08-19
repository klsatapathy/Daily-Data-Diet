# Day 3 — Variables & Data Types in Python

**Category:** Python Pantry · **Difficulty:** Beginner · **Tags:** python-pantry, variables

---

## Definition
In Python, a variable is an abstract reference (pointer) bound to an object stored in memory. Python is dynamically and strongly typed: data types are bound to values rather than variable names at runtime, and operations between incompatible types require explicit conversion.

## Rule of Thumb
Variables are labeled references to objects in memory, not fixed storage containers.

## Technical Comparison

| Data Type | Class Name | Mutability | Sequence / Mapping | Primary Data Role Use Case |
| :--- | :--- | :--- | :--- | :--- |
| Integer | `int` | Immutable | Scalar | Counts, indices, discrete metrics |
| Float | `float` | Immutable | Scalar | Continuous measurements, probabilities |
| String | `str` | Immutable | Sequence | Categorical text, IDs, raw log records |
| Boolean | `bool` | Immutable | Scalar | Conditional filtering, flags, masks |
| List | `list` | Mutable | Sequence | Dynamic collections of records/features |
| Tuple | `tuple` | Immutable | Sequence | Fixed configurations, dictionary keys, coordinates |
| Dictionary | `dict` | Mutable | Mapping | Key-value records, JSON representations, lookup tables |
| Set | `set` | Mutable | Set | Deduplication, membership testing |

## Code Example

```python
# Variable assignment and dynamic reassignment
record_id: int = 1042
print(f"Type: {type(record_id)}, Memory ID: {id(record_id)}")

# Rebinding variable to a different type
record_id: str = "ID-1042"
print(f"New Type: {type(record_id)}, New Memory ID: {id(record_id)}")

# Mutability demonstration
feature_list = [0.12, 0.45, 0.78]
initial_id = id(feature_list)
feature_list.append(0.99)  # Modified in-place
print(f"List mutated: {id(feature_list) == initial_id}")  # Returns True

feature_tuple = (0.12, 0.45, 0.78)
# feature_tuple[0] = 0.50  # Raises TypeError: tuple does not support item assignment

# Type checking best practice
if isinstance(feature_list, list):
    normalized = [x / sum(feature_list) for x in feature_list]
```

## Best Practices
* Use PEP 8 naming conventions: `snake_case` for variables, `UPPER_CASE` for constants.
* Implement type hints (`typing` module or built-in generic types) to document expected data structures for pipelines and functions.
* Validate object types using `isinstance(obj, Type)` rather than direct type comparison (`type(obj) == Type`) to support inheritance polymorphism.
* Prefer immutable types (`tuple`, `frozenset`) for fixed schemas or hashable keys to avoid accidental in-place modifications.
* Explicitly cast data types before operations (e.g., converting parsed string representations to `int` or `float`) to handle missing or malformed values predictably.

## Common Interview Question
**Q:** What is the difference between mutable and immutable data types in Python, and how does this affect function arguments?  
**A:** Mutable objects (e.g., `list`, `dict`, `set`) can have their internal state altered in-place without changing their memory address (`id`). Immutable objects (e.g., `int`, `float`, `str`, `tuple`) cannot be changed after creation; any modification returns a new object at a new memory location. Because Python passes arguments via "assignment" (call-by-object-reference), passing a mutable object allows a function to modify the caller's copy in-place, whereas operations on immutable arguments rebind the local reference without affecting the caller.

## Summary
Python variables act as dynamic references to strongly typed objects in memory. Built-in types are categorized by their scalar, sequence, or mapping behavior, as well as their mutability. Mastery of type behavior and memory referencing is critical for writing deterministic data processing pipelines and avoiding unintended state mutations.
