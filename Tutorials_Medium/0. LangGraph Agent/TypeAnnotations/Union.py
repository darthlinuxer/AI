# Type Annotation: Union
# Summary: `Union[Type1, Type2, ...]` indicates that a variable can be one of several specified types.
# Why/When: Use Union when a variable or a function parameter can accept values of different, but known, types.
# Example: `x: Union[int, float]` means `x` can be either an integer or a float.
from typing import Union

def square(x: Union[int, float]) -> float:
    if not isinstance(x, (int, float)):
        raise TypeError(f"Argument x must be int or float, got {type(x).__name__}")
    return x*x

try:
    print(square(x=5)) #ok
    print(square(x=1.234)) #ok
    x="hello"
    print(square(x)) #compiler complains but executes
except Exception as e:
    print(e)