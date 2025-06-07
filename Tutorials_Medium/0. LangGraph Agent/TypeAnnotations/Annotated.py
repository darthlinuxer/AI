# Type Annotation: Annotated
# Summary: Annotated[T, x] is used to decorate a type T with context-specific metadata x (or multiple pieces of metadata).
# Why/When: Use Annotated when you want to add extra information to a type hint that might be used by type checkers or other tools, without changing the type itself. This metadata is not used by Python at runtime for type checking directly but can be introspected.
# Example: `name: Annotated[str, 'user name']` indicates 'name' is a string and provides 'user name' as metadata.
from typing import Annotated

def greet(name: Annotated[str, 'user name']) -> str:
    return f"Hello, {name}!"

# Usage
print(greet('Alice'))
print(greet(1)) #compiler complains but executes
