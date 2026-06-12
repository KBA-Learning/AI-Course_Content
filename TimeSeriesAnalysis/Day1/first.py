import matplotlib.pyplot as plt

# 1. Define two points: (X1, Y1) and (X2, Y2)
# This will draw a line from coordinate (0, 2) straight to (10, 12)
X = [0, 10]
Y = [2, 12]

# 2. Plot the line and data points
plt.plot(X, Y, color="blue", marker="o", linewidth=2)

# 3. Add titles and show it
plt.title("Simple Line Plot")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.grid(True)

plt.show()