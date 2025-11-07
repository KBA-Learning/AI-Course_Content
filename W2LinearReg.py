import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#   Load sample dataset
# we are training the model to predict whether a customer will churn (1) or stay (0)
# based on their usage and contract details.
# Tenure	How many months the customer has been with the company
# MonthlyCharges		Monthly bill amount
# InternetService_Fiber	Whether the customer uses fiber internet
# Contract_TwoYear	Whether the customer has a long-term (2-year) contract
# Churn		Whether the customer has left (churned) or stayed
# (You can replace this with pd.read_csv("TelcoChurn.csv"))
#Prediction Goal:
# Given customer features (like tenure, monthly charges, and contract type),
# can we predict if they are likely to churn (leave) or stay with the telecom company?
data = {
    'Tenure': [1, 3, 5, 10, 2, 8, 12, 15, 20, 25],
    'MonthlyCharges': [70, 50, 80, 60, 90, 55, 100, 65, 40, 30],
    'InternetService_Fiber': [1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
    'Contract_TwoYear': [0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
    'Churn': [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
}
#builds a structured table from the data.
df = pd.DataFrame(data)

print("Dataset preview:")
print(df.head(), "\n")

# Define features (X) and target (y)
X = df[['Tenure', 'MonthlyCharges', 'InternetService_Fiber', 'Contract_TwoYear']]
y = df['Churn']

# Split data into train and test sets
# 30% of the data will go to testing, 70% to training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Train size:", len(X_train), " Test size:", len(X_test), "\n")

# Initialize and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

#  Make predictions
y_pred = model.predict(X_test)

# ✅ Step 7: Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("🎯 Accuracy:", round(accuracy * 100, 2), "%\n")

print("🔹 Classification Report:")
print(classification_report(y_test, y_pred))

print("🔹 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
