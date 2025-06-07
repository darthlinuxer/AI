# Type Annotation: NewType
# Summary: `NewType('TypeName', BaseType)` is used to create distinct types that are subtypes of an existing type for the purpose of type checking, but behave like the base type at runtime.
# Why/When: Use NewType to create semantically different types that share the same underlying representation (e.g., UserId and OrderId both being integers) to prevent accidental misuse. Type checkers will treat them as incompatible.
# Example: `UserId = NewType('UserId', int)` creates a new type `UserId` that is a distinct type from `int` for static analysis.
from typing import NewType

UserId = NewType('UserId', int)

def get_user_name(user_id: UserId) -> str:
    return f"User{user_id}"

# Usage
uid = UserId(123)
print(get_user_name(uid))  # User123
