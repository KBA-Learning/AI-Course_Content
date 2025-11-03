import numpy as np

# Create arrays
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print("Sum:", a + b)
print("Mean:", a.mean())
print("Shape:", a.shape)   # tells the dimensions of the array. 

# 2D array
matrix = np.array([[1, 2], [3, 4]])
print(matrix[0, 1])  # Access element

import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Age": [25, 30, 35, None],
    "Score": [85, 92, 88, 90]
}

df = pd.DataFrame(data)
print(df.head())
print(df["Age"].mean())       # Handling missing data

print("\n--- HEAD ---")
print(df.head())  # View first few rows

print("\n--- INFO ---")
print(df.info())  # Data types, null counts

print("\n--- DESCRIBE ---")
print(df.describe())  # Summary statistics

print("\nAverage Age (with NaN handled):", df["Age"].mean())

# --- Filter example ---
print("\n--- FILTER: Scores > 88 ---")
print(df[df["Score"] > 88])

# --- GroupBy example ---
print("\n--- GROUPBY on 'Score' ---")
print(df.groupby("Score").count())

# --- Fill missing values ---
df["Age"].fillna(df["Age"].mean(), inplace=True)
print("\n--- AFTER FILLING MISSING AGE ---")
print(df)

# Plotting a graph

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, label='Sine Wave')
plt.title("Sine Function")
plt.xlabel("x values")
plt.ylabel("sin(x)")
plt.legend()
plt.grid(True)

# Save figure
plt.savefig("sine_plot.png")
plt.show()
