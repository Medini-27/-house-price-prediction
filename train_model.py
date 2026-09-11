import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "data/housing.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("HOUSE PRICE PREDICTION - MODEL TRAINING")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")

print("\nFirst 5 records:")
print(df.head())


# ============================================================
# 2. CHECK DATA
# ============================================================

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# Remove duplicates
df = df.drop_duplicates()


# ============================================================
# 3. DEFINE FEATURES AND TARGET
# ============================================================

features = [
    "City",
    "Locality",
    "BHK",
    "Area_SqFt",
    "Bathrooms",
    "Balconies",
    "Furnishing",
    "Property_Age_Yrs"
]

target = "Price_Lakhs_INR"

X = df[features]
y = df[target]


# ============================================================
# 4. IDENTIFY COLUMN TYPES
# ============================================================

categorical_features = [
    "City",
    "Locality",
    "Furnishing"
]

numeric_features = [
    "BHK",
    "Area_SqFt",
    "Bathrooms",
    "Balconies",
    "Property_Age_Yrs"
]


# ============================================================
# 5. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 7. RANDOM FOREST MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 8. CREATE COMPLETE ML PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# 9. TRAIN
# ============================================================

print("\nTraining Random Forest model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed!")


# ============================================================
# 10. PREDICT
# ============================================================

y_pred = pipeline.predict(X_test)


# ============================================================
# 11. EVALUATE
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE  : ₹{mae:.2f} Lakhs")
print(f"RMSE : ₹{rmse:.2f} Lakhs")
print(f"R²   : {r2:.4f}")

print("=" * 60)


# ============================================================
# 12. SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

model_path = "models/house_price_model.pkl"

joblib.dump(
    pipeline,
    model_path
)

print(f"\nModel saved successfully:")
print(model_path)


# ============================================================
# 13. SAVE METRICS
# ============================================================

metrics = {
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}

joblib.dump(
    metrics,
    "models/metrics.pkl"
)

print("Metrics saved successfully.")

print("\nDone! Model training completed.")
