import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
df = pd.read_csv('daily_retail_sales.csv') 
# 1. Feature Engineering (Data Processing)
df['Lag_1'] = df['Units_Sold'].shift(1)
df['Lag_7'] = df['Units_Sold'].shift(7)  # Previous week's sales feature
df['Rolling_Mean_7'] = df['Units_Sold'].shift(1).rolling(window=7).mean()
df = df.dropna()  # Cleaning out the NaN rows

# 2. Chronological Split
train = df[df['Date'] < '2026-05-25']
test = df[df['Date'] >= '2026-05-25']

X_train = train[['Lag_1', 'Lag_7', 'Rolling_Mean_7']]
y_train = train['Units_Sold']
X_test = test[['Lag_1', 'Lag_7', 'Rolling_Mean_7']]
y_test = test['Units_Sold']

# 3. Training our Linear Machine
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

print(f"Linear Regression MAE: {mean_absolute_error(y_test, lr_preds):.2f}")