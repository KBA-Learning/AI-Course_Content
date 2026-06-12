import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. CREATE THE SIMPLIFIED DATASET
# ==========================================
# Simple pizza size vs price data from Day 1 notes
data = {
    "Size_Inches": [6, 8, 10, 12, 14],
    "Price_Dollars": [8, 11, 14, 17, 20],
}
df = pd.DataFrame(data)

X = df[["Size_Inches"]]
y = df["Price_Dollars"]

# ==========================================
# 2. FIND THE LINE OF BEST FIT
# ==========================================
model = LinearRegression()
model.fit(X, y)

# FIX: Generate the line array, but wrap it in a DataFrame with matching column names
raw_array = np.linspace(5, 15, 100)
x_line = pd.DataFrame({"Size_Inches": raw_array})

# Now predict using the DataFrame—no more warnings!
y_line = model.predict(x_line)

# ==========================================
# 3. PLOT THE DATA POINTS & THE LINE
# ==========================================
# Plot the actual collected data points as red dots
plt.scatter(
    df["Size_Inches"],
    df["Price_Dollars"],
    color="red",
    s=100,
    label="Actual Data Points",
)

# Plot the calculated linear regression line passing through them
plt.plot(
    x_line,
    y_line,
    color="darkblue",
    linewidth=2,
    label="Line of Best Fit ($Y = WX + B$)",
)

# Add clear chart titles and labels
plt.title("Linear Regression: Visualizing the Line of Best Fit", fontsize=14)
plt.xlabel("Pizza Size (Inches)", fontsize=12)
plt.ylabel("Price (Dollars)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
# Save or display the finalized plot
plt.savefig("pizza_regression.png")