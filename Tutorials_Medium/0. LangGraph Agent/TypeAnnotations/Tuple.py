# Type Annotation: Tuple
# Summary: Tuple[int, int] is used to specify a fixed-size sequence of elements with known types (here, two integers).
# Why/When: Use Tuple when a function returns or accepts a fixed number of elements of possibly different types, ensuring type safety and clarity.
# Example: get_point() -> Tuple[int, int] returns a pair of integers representing a point.
from typing import Tuple

def get_point() -> Tuple[int, int]:
    return (10, 20)

# Usage
x, y = get_point()
print(x, y)  # 10 20
