import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
file_path = r"c:\Users\dell\adp_pbl\shelter_dataset_all_updated.csv"
df = pd.read_csv(file_path)

# Select features and target
features = [
    'total_capacity', 'average_age', 'male_percentage', 'female_percentage',
    'temperature', 'rainfall_mm', 'area_population', 'staff_count',
    'children_count', 'senior_citizens_count', 'new_admissions', 'exits_today',
    'emergency_cases', 'unemployment_rate', 'crime_rate',
    'season', 'city', 'funding_level', 'day_of_week'
]
target = 'occupied_beds'

X = df[features]
y = df[target]

# Preprocessing for categorical data
categorical_features = ['season', 'city', 'funding_level', 'day_of_week']
numerical_features = [col for col in features if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Define the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training model...")
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Training Complete.")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# Save the model
joblib.dump(model, 'model.pkl')
print("Model saved as model.pkl")
