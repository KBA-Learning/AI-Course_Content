import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# ==========================================
# 1. CREATE DATA & SET THE TIME INDEX
# ==========================================
np.random.seed(42)
dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
sales = 50 + np.sin(np.linspace(0, 15, 100)) * 10 + np.random.normal(0, 3, 100)

df = pd.DataFrame({"Actual_Sales": sales}, index=dates)

# Split into Train (first 80) and Test (last 20)
train_df = df.iloc[:80]
test_df = df.iloc[80:].copy()

# ==========================================
# 2. THE MACHINE LEARNING WORKFLOW (.fit())
# ==========================================
# We initialize the ARIMA model on our training data.
# The order=(0, 0, 3) tells the model: "Act exactly like a 3-day Moving Average"
model = ARIMA(train_df["Actual_Sales"], order=(0, 0, 3))

# STEP A: STUDY/FIT (The model analyzes the training data history)
model_fitted = model.fit()

# ==========================================
# 3. THE PREDICTION WORKFLOW (.predict())
# ==========================================
# STEP B: PREDICT (We tell it exactly which dates we want to forecast)
predictions = model_fitted.predict(start=test_df.index[0], end=test_df.index[-1])

# Save guesses back to our test table
test_df["Predicted_Sales"] = predictions

# ==========================================
# 4. PLOT THE RESULTS
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(train_df.index, train_df["Actual_Sales"], label="Train Data", color="blue")
plt.plot(test_df.index, test_df["Actual_Sales"], label="Actual Future", color="green")
plt.plot(test_df.index, test_df["Predicted_Sales"], label="ARIMA Fit/Predict", color="red", linestyle="--")
plt.title("Moving Average Forecasting using .fit() and .predict()")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()