import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 1. CREATE SIMULATED BAKERY SALES DATA
# ==========================================
np.random.seed(42)
dates = pd.date_range(start="2026-01-01", periods=100)
sales = 50 + np.sin(np.linspace(0, 15, 100)) * 10 + np.random.normal(0, 3, 100)

df = pd.DataFrame({"Actual_Sales": sales}, index=dates)

print("--- First 5 rows of raw data ---")
print(df.head())
print("\n" + "=" * 40 + "\n")

# ==========================================
# 2. INBUILT FEATURE ENGINEERING (THE FORECAST)
# ==========================================
# window_size = 3 (Look back at the last 3 days)
# .shift(1) ensures we use YESTERDAY'S back-data to predict TODAY
df["Predicted_Sales"] = (
    df["Actual_Sales"].shift(1).rolling(window=3).mean()
)

# ==========================================
# 3. CHRONOLOGICAL TRAIN / TEST SPLIT
# ==========================================
# We split exactly at row 80 to evaluate our predictions on the final 20 days
split_point = 80
train_df = df.iloc[:split_point]

# For the test set, we drop the first row of the test set if it contains any NaN,
# but using shift(1).rolling(3) means NaNs only happen at the very beginning of the dataset (first 3 rows).
test_df = df.iloc[split_point:].copy()

# ==========================================
# 4. ERROR EVALUATION (MAE)
# ==========================================
test_df["Error"] = np.abs(
    test_df["Actual_Sales"] - test_df["Predicted_Sales"]
)
mae = test_df["Error"].mean()

print("--- Test Set Predictions (Inbuilt Pandas) ---")
print(test_df[["Actual_Sales", "Predicted_Sales"]].head())
print("\n" + "=" * 40 + "\n")

print(f"Our model's Mean Absolute Error: {mae:.2f} loaves")
print(
    f"This means on average, our daily guesses were off by about {round(mae)} loaves."
)

# ==========================================
# 5. VISUALIZE THE RESULTS
# ==========================================
plt.figure(figsize=(12, 6))

# Plot the past training data
plt.plot(
    train_df.index,
    train_df["Actual_Sales"],
    label="Past Data (Train)",
    color="blue",
)

# Plot the actual "future" data
plt.plot(
    test_df.index,
    test_df["Actual_Sales"],
    label="Real Future Sales (Test Actuals)",
    color="green",
    marker="o",
)

# Plot our engineered moving average guesses
plt.plot(
    test_df.index,
    test_df["Predicted_Sales"],
    label="Inbuilt Moving Avg Guesses (Predictions)",
    color="red",
    linestyle="--",
    marker="x",
)

plt.title("Bakery Sales: Inbuilt Pandas Rolling Forecast")
plt.xlabel("Date")
plt.ylabel("Loaves of Bread Sold")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

# Render the plot
plt.show()