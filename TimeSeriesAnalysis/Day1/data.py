import pandas as pd
import numpy as np
# 1. Generate a sequence of daily dates for 2 years
dates = pd.date_range(start="2024-01-01", end="2026-06-01", freq="D")
# 2. Create mock sales data with an upward trend and weekend spikes
base_sales = np.linspace(100, 250, len(dates))  # General upward trend
weekly_pattern = np.array([0, 5, 10, 8, 15, 40, 50])[dates.dayofweek]  # Higher sales on weekends
random_noise = np.random.normal(0, 10, len(dates))  # Random daily fluctuations
units_sold = (base_sales + weekly_pattern + random_noise).astype(int)
# 3. Combine into a spreadsheet and save it
mock_df = pd.DataFrame({"Date": dates, "Units_Sold": units_sold})
mock_df.to_csv("daily_retail_sales.csv", index=False)
print("Success! 'daily_retail_sales.csv' has been created in your folder.")
