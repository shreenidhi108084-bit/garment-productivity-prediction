import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ==========================
# Load Dataset
# ==========================
data = pd.read_csv("garments_worker_productivity.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset Shape:", data.shape)

# ==========================
# Check Null Values
# ==========================
print("\nMissing Values:")
print(data.isnull().sum())

# Fill missing values in WIP column
data['wip'] = data['wip'].fillna(data['wip'].mean())

# ==========================
# Remove Date Column
# ==========================
data.drop('date', axis=1, inplace=True)

# ==========================
# Encode Categorical Columns
# ==========================
encoder = LabelEncoder()

data['quarter'] = encoder.fit_transform(data['quarter'])
data['department'] = encoder.fit_transform(data['department'])
data['day'] = encoder.fit_transform(data['day'])

# ==========================
# Split Features and Target
# ==========================
X = data.drop('actual_productivity', axis=1)
y = data['actual_productivity']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Train XGBoost Model
# ==========================
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# Predictions
# ==========================
pred = model.predict(X_test)

# ==========================
# Evaluation
# ==========================
print("\nModel Performance")

print("MAE =", mean_absolute_error(y_test, pred))
print("MSE =", mean_squared_error(y_test, pred))
print("R2 Score =", r2_score(y_test, pred))

# ==========================
# Save Model
# ==========================
pickle.dump(model, open("model.pkl", "wb"))

print("\nModel saved as model.pkl")

# ==========================
# User Prediction
# ==========================
print("\nEnter Values for Prediction")

quarter = int(input("Quarter (0-4): "))
department = int(input("Department (0-1): "))
day = int(input("Day (0-5): "))
team = int(input("Team: "))
targeted_productivity = float(input("Targeted Productivity: "))
smv = float(input("SMV: "))
wip = float(input("WIP: "))
over_time = int(input("Over Time: "))
incentive = int(input("Incentive: "))
idle_time = float(input("Idle Time: "))
idle_men = int(input("Idle Men: "))
no_of_style_change = int(input("Number of Style Changes: "))
no_of_workers = float(input("Number of Workers: "))

sample = np.array([[
    quarter,
    department,
    day,
    team,
    targeted_productivity,
    smv,
    wip,
    over_time,
    incentive,
    idle_time,
    idle_men,
    no_of_style_change,
    no_of_workers
]])

prediction = model.predict(sample)

print("\nPredicted Productivity =", round(prediction[0], 4))
