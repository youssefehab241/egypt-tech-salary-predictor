from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = Path("data_sources/merged/final_model_training_data.csv")
MODEL_DIR = Path("salary_prediction_model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "final_salary_model.pkl"
METRICS_PATH = MODEL_DIR / "final_model_metrics.json"


def remove_outliers_per_group(df, group_col="job_title", target_col="salary_mid"):
    kept_parts = []

    for group_name, group in df.groupby(group_col):
        q1 = group[target_col].quantile(0.25)
        q3 = group[target_col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        filtered = group[(group[target_col] >= lower) & (group[target_col] <= upper)]
        kept_parts.append(filtered)

    return pd.concat(kept_parts, ignore_index=True)


# =========================
# Load data
# =========================
df = pd.read_csv(DATA_PATH)

print("Initial shape:", df.shape)
print(df.head())

required_columns = ["job_title", "experience_years", "salary_mid"]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

df = df[required_columns].copy()

df["job_title"] = df["job_title"].astype(str).str.strip().str.lower()
df["experience_years"] = pd.to_numeric(df["experience_years"], errors="coerce")
df["salary_mid"] = pd.to_numeric(df["salary_mid"], errors="coerce")

df = df.dropna(subset=["job_title", "experience_years", "salary_mid"])
df = df[df["job_title"] != ""]
df = df[df["salary_mid"] >= 1000]
df = df[df["experience_years"] >= 0]

print("\nShape before outlier removal:", df.shape)

df = remove_outliers_per_group(df, group_col="job_title", target_col="salary_mid")

print("Shape after outlier removal:", df.shape)
print("\nJob title counts:")
print(df["job_title"].value_counts())

X = df[["job_title", "experience_years"]]
y = df["salary_mid"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

categorical_features = ["job_title"]
numeric_features = ["experience_years"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        ),
        (
            "num",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            numeric_features
        )
    ]
)

base_models = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

results = {}
best_model_name = None
best_pipeline = None
best_mae = float("inf")

for model_name, model in base_models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1
        ))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    results[model_name] = {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "R2": round(float(r2), 4)
    }

    print(f"\n=== {model_name} ===")
    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 4))

    if mae < best_mae:
        best_mae = mae
        best_model_name = model_name
        best_pipeline = pipeline

joblib.dump(best_pipeline, BEST_MODEL_PATH)

summary = {
    "best_model": best_model_name,
    "metrics": results,
    "dataset_shape_after_cleaning": {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1])
    },
    "features": ["job_title", "experience_years"],
    "target": "salary_mid"
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print("\nBest model:", best_model_name)
print("Saved model to:", BEST_MODEL_PATH)
print("Saved metrics to:", METRICS_PATH)