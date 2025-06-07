# Type Annotation: Final
# Summary: `Final` indicates that a variable or attribute should not be reassigned after its initial assignment. It's used to declare constants.
# Why/When: Use Final to define constants to prevent accidental modification. Type checkers will issue a warning or error if a Final variable is reassigned.
# Example: `PI: Final = 3.14159` declares PI as a constant float that should not be changed.
from typing import Final

PI: Final = 3.14159

# PI = 3  # This will be flagged by type checkers as an error
print(PI)
