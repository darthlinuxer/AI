# Type Annotation: Callable
# Summary: Callable[[Arg1Type, Arg2Type, ...], ReturnType] is used to specify the signature of a callable object, like a function or a method.
# Why/When: Use Callable when a function accepts another function (or any callable) as an argument or returns one, to ensure the passed/returned callable matches the expected signature.
# Example: `f: Callable[[int, int], int]` means `f` is a function that takes two integers as arguments and returns an integer.
from typing import Callable

def apply_func(f: Callable[[int, int], int], a: int, b: int) -> int:
    return f(a, b)

# Usage
def add(x: int, y: int) -> int:
    return x + y

print(apply_func(add, 2, 3))  # 5
