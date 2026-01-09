import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# 1. Load the dataset
data = pd.read_csv("dataset/winequality-red.csv", sep=";")

# 2. Feature selection
X = data.drop("quality", axis=1)
y = data["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Pre-processing (scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Train the model (Ridge Regression)
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# 4. Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 6. Print metrics to standard output
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# 5. Save trained model
joblib.dump(model, "output/model.pkl")

# 5. Save evaluation metrics to JSON
results = {
    "MSE": mse,
    "R2": r2
}

with open("output/results.json", "w") as f:
    json.dump(results, f, indent=4)
