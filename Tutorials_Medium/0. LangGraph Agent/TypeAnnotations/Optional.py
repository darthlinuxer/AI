# Type Annotation: Optional
# Summary: `Optional[T]` is equivalent to `Union[T, None]`. It indicates that a variable can either be of type T or be `None`.
# Why/When: Use Optional for parameters that can be omitted (often with a default value of `None`) or for variables that might not have a value at a certain point.
# Example: `name: Optional[str] = None` means `name` can be a string or `None`.
from typing import Optional

def nice_message(name: Optional[str] = None) -> None:
    if name is None:
        print("Hey random person")
        return
    print(f"Hey {name}")
    
nice_message()
nice_message("Bob")