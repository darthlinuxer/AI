# Type Annotation: Dict
# Summary: `Dict[KeyType, ValueType]` is used to specify the types for keys and values within a dictionary.
# Why/When: Use Dict to ensure that a dictionary adheres to a specific structure, improving code readability and preventing runtime errors due to incorrect key or value types.
# Example: `people: Dict[str, int]` means `people` is a dictionary where keys are strings and values are integers.
from typing import Dict

def get_age(people: Dict[str, int], name: str) -> int:
    return people.get(name, -1)

# Usage
ages = {'Alice': 30, 'Bob': 25}
print(get_age(ages, 'Alice'))  # 30
