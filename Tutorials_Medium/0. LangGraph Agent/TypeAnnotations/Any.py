# Type Annotation: Any
# Summary: `Any` is a special type indicating an unconstrained type. A variable annotated with `Any` can accept a value of any type.
# Why/When: Use `Any` when it's not possible to specify a more precise type, or when you are gradually migrating a codebase to static typing and want to mark parts that are not yet fully typed. It effectively disables type checking for that specific part.
# Example: `x: Any` means `x` can be an integer, string, list, etc.
from typing import Any

def print_value(x:Any) -> None:
    print(x)
    
print_value(1)