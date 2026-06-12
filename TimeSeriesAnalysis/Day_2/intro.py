import matplotlib.pyplot as plt


import pandas as pd


# Step 1: Build a synthetic 3-week dataset with a weekend spike pattern
data = {
   "Date": pd.date_range(start="2026-05-01", periods=21, freq="D"),
   "Sales": [
       100,
       105,
       98,
       110,
       165,
       195,
       180,  # Week 1: High weekend demand (Fri/Sat/Sun)
       105,
       103,
       108,
       115,
       170,
       210,
       192,  # Week 2: Same repeating pattern
       112,
       109,
       114,
       122,
       180,
       225,
       205,  # Week 3: Growth trend begins to show slightly
   ],
}
df = pd.DataFrame(data)


# Step 2: Crucial step - Transform the Date column into the DataFrame Index
# This changes the dataset structure from a standard table into a Time Series.
df.set_index("Date", inplace=True)


# Step 3: Generate the visual time series chart
plt.figure(figsize=(11, 5))
plt.plot(df.index, df["Sales"], marker="o", color="darkblue", linestyle="-")


# Step 4: Add explicit context and labels for business presentations
plt.title("Product Demand Analysis: Daily Milk Sales Pattern", fontsize=14)
plt.xlabel("Timeline (Days)", fontsize=12)
plt.ylabel("Units Sold (Cartons)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
print(df.head())
print(df.dtypes)
# Step 5: Render the plot on screen
plt.show()
