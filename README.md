# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("path_to_dataset.csv")  # Replace with the actual dataset path

# Display dataset information
print("Dataset Preview:\n", df.head())
print("\nDataset Info:\n")
df.info()

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Handle missing values (if any)
df = df.dropna()  # Remove rows with missing values (or use imputation)

# Check dataset statistics
print("\nDataset Statistics:\n", df.describe())

# Correlation heatmap to understand feature relationships
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Feature selection (assuming 'Price' is the target column)
X = df.drop(columns=['Price'])  # Replace 'Price' with the actual target column name
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared Score: {r2}")

# Feature importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:\n", feature_importances)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette="viridis")
plt.title("Feature Importance")
plt.show()

# Save the trained model (optional)
import joblib
joblib.dump(model, "car_price_model.pkl")
print("\nTrained model saved as 'car_price_model.pkl'")
