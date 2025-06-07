# Type Annotation: TypedDict
# Summary: `TypedDict` is used to define dictionary types with a fixed set of string keys and specific value types for each key.
# Why/When: Use TypedDict when you need to define the structure of dictionaries that are used like records or objects with fixed fields, allowing type checkers to verify key presence and value types.
# Example: `class Movie(TypedDict): title: str; release_year: int` defines a dictionary structure for movie data.
from typing import TypedDict

class Movie(TypedDict):
    title: str
    release_year: int
    director: str
    rating: float

movie = {
    "title": "The Shawshank Redemption",
    "release_year": 1994,
    "director": "Frank Darabont",
    "rating": 9.3
}

print(movie)