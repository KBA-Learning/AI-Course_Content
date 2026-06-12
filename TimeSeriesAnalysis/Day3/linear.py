import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ==========================================
# 1. LOAD DATA & CONVERT DATES
# ==========================================
df = pd.read_csv("daily_retail_sales.csv")

# Ensure the Date column is recognized as actual dates for accurate plotting
df["Date"] = pd.to_datetime(df["Date"])

# ==========================================
# 2. FEATURE ENGINEERING (DATA PROCESSING)
# ==========================================
df["Lag_1"] = df["Units_Sold"].shift(1)
df["Lag_7"] = df["Units_Sold"].shift(7)  # Previous week's sales feature
df["Rolling_Mean_7"] = (
    df["Units_Sold"].shift(1).rolling(window=7).mean()
)
df = df.dropna()  # Cleaning out the NaN rows

# ==========================================
# 3. CHRONOLOGICAL SPLIT
# ==========================================
# Make sure comparison uses matching datetime formats
split_date = pd.to_datetime("2026-05-25")
train = df[df["Date"] < split_date]
test = df[df["Date"] >= split_date]

X_train = train[["Lag_1", "Lag_7", "Rolling_Mean_7"]]
y_train = train["Units_Sold"]
X_test = test[["Lag_1", "Lag_7", "Rolling_Mean_7"]]
y_test = test["Units_Sold"]

# ==========================================
# 4. TRAINING OUR LINEAR MACHINE
# ==========================================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

print(f"Linear Regression MAE: {mean_absolute_error(y_test, lr_preds):.2f}")

# ==========================================
# 5. VISUALIZE THE TIME SERIES RESULTS
# ==========================================
# Adjust size to make sure dates are fully readable and labels aren't overlapping
plt.figure(figsize=(12, 6))

# 1. Plot a small window of recent history (e.g., last 30 days of training data)
# This provides visual context without overcrowding the chart with 2 years of lines
recent_train = train.tail(30)
plt.plot(
    recent_train["Date"],
    recent_train["Units_Sold"],
    label="Past Sales (Train Window)",
    color="blue",
    alpha=0.6,
)

# 2. Plot the real actual future sales in the testing window
plt.plot(
    test["Date"],
    y_test,
    label="Real Future Sales (Test Actuals)",
    color="green",
    marker="o",
)

# 3. Plot the model's predictions over the same testing window
plt.plot(
    test["Date"],
    lr_preds,
    label="Linear Regression Forecast",
    color="red",
    linestyle="--",
    marker="x",
)

# 4. Draw a distinct vertical split boundary line
plt.axvline(
    split_date,
    color="black",
    linestyle=":",
    linewidth=2,
    label="Forecast Start Cutoff",
)

# Aesthetics & Labels
plt.title(
    "Retail Demand Forecasting: Linear Regression with Lag Features",
    fontsize=14,
)
plt.xlabel("Timeline (Dates)", fontsize=12)
plt.ylabel("Units Sold", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left")

# Rotate date ticks on x-axis slightly so they remain non-overlapping
plt.xticks(rotation=15)
plt.tight_layout()

# Render/Save the plot
plt.show()