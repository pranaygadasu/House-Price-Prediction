# train.py
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -----------------------------
# 1. Load dataset safely
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # project root
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")

df = pd.read_csv(DATA_PATH)

# Create a feature: house age
df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

# Target variable
y = df["SalePrice"]
X = df.drop(columns=["SalePrice", "Id"])

# -----------------------------
# 2. Preprocessing
# -----------------------------
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -----------------------------
# 3. Model pipeline
# -----------------------------
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# -----------------------------
# 4. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

# -----------------------------
# 5. Evaluation
# -----------------------------
y_pred = pipeline.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# -----------------------------
# 6. Save model + metadata
# -----------------------------
MODEL_PATH = os.path.join(BASE_DIR, "models", "house_price_pipeline.pkl")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

artifact = {
    "pipeline": pipeline,
    "feature_names": list(X.columns),             # EXACT feature columns used during training
    "numeric_features": numeric_features,
    "categorical_features": categorical_features
}

joblib.dump(artifact, MODEL_PATH)
print(f"✅ Model + metadata saved to {MODEL_PATH}")