import pandas as pd
import numpy as np

# Creating DataFrame
data = {
    'School ID': [101, 102, 103, np.nan, 105, 106, 107, 108],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Grace', 'Henry'],
    'Address': ['123 Main St', '456 Oak Ave', '789 Pine Ln', '101 Elm St', np.nan, '222 Maple Rd', '444 Cedar Blvd', '555 Birch Dr'],
    'City': ['Los Angeles', 'New York', 'Houston', 'Los Angeles', 'Miami', np.nan, 'Houston', 'New York'],
    'Subject': ['Math', 'English', 'Science', 'Math', 'History', 'Math', 'Science', 'English'],
    'Marks': [85, 92, 78, 89, np.nan, 95, 80, 88],
    'Rank': [2, 1, 4, 3, 8, 1, 5, 3],
    'Grade': ['B', 'A', 'C', 'B', 'D', 'A', 'C', 'B']
}

df= pd.DataFrame(data)
print("Data Frame:",df)

# Handling Missing Data

# 1.Removing rows with Missing values
df_cleaned = df.dropna()
print("Cleaned data after dropping")

print(df_cleaned)

# 2.Mean,Mode,Median Imputation

mean_imputation = df['Marks'].fillna(df['Marks'].mean())   #average
print('Mean Imputation')
print(mean_imputation)

mode_v = df['Marks'].mode().min()
print(mode_v)
mode_imputation = df['Marks'].fillna(df['Marks'].mode().min())
print('Mode Imputation')
print(mode_imputation)
mode_imputation = df['Marks'].fillna(df['Marks'].mode().iloc[0])  # sort in ascending order and get the first value
print('Mode Imputation')
print(mode_imputation)

median_imputation = df['Marks'].fillna(df['Marks'].median())   # middle value
print('Median Implication')
print(median_imputation)

# Forward fill
forwardfill = df['Marks'].fillna(method='ffill')
print('Forward Fill')
print(forwardfill)

# Backward Fill
backwardfill = df['Marks'].fillna(method='bfill')
print('backwardfill')
print(backwardfill)

# Forward Fill
forwardfill = df['Marks'].fillna(method='ffill')
print('Forward fill')
print(forwardfill)