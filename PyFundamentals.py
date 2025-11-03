#display statement
print("Hello, world! Welcome to Python.")

#variables and Data Types
name = "Alice"
age = 18
print("Name:", name)
print("Age:", age)

# Input and Output
user_name = input("Enter your name: ")
print("Nice to meet you,", user_name)

# add two numbers and display sum
a = int(input("Enter first number: "))
b = int(input("Enter second number: "))
print("Sum:", a + b)


# DAY-2
x = 10
y = 3.14
name = "Alice"
is_valid = True

# list tuples, dictionary, set
fruits = ["apple", "banana", "cherry"]
print(fruits[0])          # Indexing
print(fruits[-1])         # Last item
print(fruits[1:])         # Slicing

person = {"name": "Eve", "age": 30}
print(person["name"])

unique_nums = {1, 2, 2, 3, 4}
print(unique_nums)

print(type(x), type(y), type(name), type(is_valid))
nums = [10, 20, 30, 40, 50]
print(nums[1:4])  # slicing
for n in nums:
    print(n)

squares = [x**2 for x in range(5)]
even_squares = [x**2 for x in range(10) if x%2==0]

#functions and control flow
def greet(name):
    return f"Hello, {name}!"

print(greet("Sumi"))


try:
    x = int(input("Enter a number: "))
    print(10 / x)
except ZeroDivisionError:
    print("Cannot divide by zero!")
except ValueError:
    print("Please enter a valid number.")

# modules, packges, virtual environment. 
import math
print(math.sqrt(16))
