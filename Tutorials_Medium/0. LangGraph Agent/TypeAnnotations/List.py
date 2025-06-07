# Type Annotation: List
# Summary: `List[ElementType]` is used to specify that a variable is a list containing elements of a specific type.
# Why/When: Use List to ensure that a list contains elements of an expected type, improving code robustness and readability.
# Example: `numbers: List[int]` means `numbers` is a list where all elements are integers.
from typing import List

def sum_numbers(numbers: List[int]) -> int:
    return sum(numbers)

# Usage
nums = [1, 2, 3]
result = sum_numbers(nums)
print(result)  # 6
