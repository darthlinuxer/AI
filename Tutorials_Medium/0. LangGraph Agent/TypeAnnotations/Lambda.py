# File Summary: Lambda Functions and Map
# This file demonstrates the use of lambda functions for creating small, anonymous functions
# and the `map()` function to apply a function to all items in an iterable.
square = lambda x: x*x
print(square(10))

nums = [1,2,3,4]
squares = list(map(square, nums))
print(squares)
