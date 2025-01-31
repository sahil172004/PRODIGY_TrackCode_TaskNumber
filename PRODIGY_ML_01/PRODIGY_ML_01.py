import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset safely
file_path = "train.csv"  # Update with the correct path
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File '{file_path}' not found. Please check the path.")

df = pd.read_csv(file_path)

# Selecting relevant features
features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
target = "SalePrice"

# Ensure the required columns exist
missing_cols = [col for col in features + [target] if col not in df.columns]
if missing_cols:
    raise KeyError(f"Missing columns in dataset: {missing_cols}")

# Handle missing values by filling with median
df[features] = df[features].fillna(df[features].median())
df[target] = df[target].fillna(df[target].median())

# Splitting the dataset
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Visualizing actual vs predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label="Predictions")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Perfect Fit")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()
