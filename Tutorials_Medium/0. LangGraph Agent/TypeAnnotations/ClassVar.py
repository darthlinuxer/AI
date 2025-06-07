# Type Annotation: ClassVar
# Summary: `ClassVar[T]` is used to mark a variable as a class variable. It indicates that the variable belongs to the class itself, not to instances of the class.
# Why/When: Use ClassVar to clearly distinguish class variables from instance variables. Type checkers will flag attempts to set a ClassVar on an instance.
# Example: `species: ClassVar[str] = "Canis familiaris"` declares `species` as a class variable of type string.
from typing import ClassVar

class Dog:
    # ClassVar is a special type construct to mark class variables.
    # An annotation wrapped in ClassVar indicates that a given
    # attribute is intended to be used as a class variable and
    # should not be set on instances of that class.
    species: ClassVar[str] = "Canis familiaris"
    def __init__(self, name: str):
        self.name = name

# Usage
print(Dog.species)
d = Dog("Fido")
print(d.name)
