# Type Annotation: Self (Python 3.11+)
# Summary: `Self` is used within a class method to refer to an instance of the class itself. It's particularly useful for methods that return `self` (e.g., in builder patterns or for chained calls) and in generic classes.
# Why/When: Use Self to accurately type methods that return an instance of their own class, especially in scenarios involving inheritance, as it correctly refers to the subclass type when inherited.
# Example: `def set_value(self, value: int) -> Self:` indicates the method returns an instance of the class it's called on.
from typing import Self

class Builder:
    def set_value(self, value: int) -> Self:
        self.value = value
        return self

# Usage
b = Builder().set_value(10)
print(b.value)
