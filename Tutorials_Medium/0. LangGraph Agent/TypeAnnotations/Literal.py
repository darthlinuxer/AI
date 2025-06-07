# Type Annotation: Literal
# Summary: `Literal[v1, v2, ...]` is used to specify that a variable can only have one of a few specific literal values.
# Why/When: Use Literal when a variable must be one of a predefined set of constant values (e.g., specific strings or numbers), enhancing type safety beyond simple type checking.
# Example: `status: Literal['success', 'error']` means `status` can only be the string 'success' or 'error'.
from typing import Literal

def get_status(code: int) -> Literal['success', 'error']:
    if code == 0:
        return 'success'
    else:
        return 'error'

# Usage
print(get_status(0))      # success
print(get_status(1))      # error
