import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# 1. LOAD & CLEAN
df = pd.read_csv("data_sources/merged/ml_ready_salary_data_standardized.csv")

df = df.drop(columns=["job_title_clean"], errors="ignore")
df = df.rename(columns={"job_title_standardized": "job_title_clean"})

# CRITICAL: Smart K-Correction
df.loc[df["salary_target"] < 200, "salary_target"] *= 1000

# Remove extreme noise
df = df[(df["salary_target"] >= 5000) & (df["salary_target"] <= 250000)].copy()

df = df[
    [
        "job_title_clean",
        "experience_years_clean",
        "location_clean",
        "work_mode_clean",
        "level_clean",
        "salary_target",
    ]
].copy()

# 2. ENCODE
X = pd.get_dummies(df.drop(columns=["salary_target"]))
X = X.astype(float)
y = np.log1p(df["salary_target"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. HYPERPARAMETER TUNING
print("Searching for best model parameters...")

param_grid = {
    "n_estimators": [500, 1000, 1500],
    "max_depth": [6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "gamma": [0, 0.1, 0.2],
    "reg_lambda": [1, 2, 5],
}

xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)

search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_grid,
    n_iter=15,
    cv=3,
    scoring="neg_mean_absolute_error",
    verbose=1,
    n_jobs=-1,
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

print(f"\nBest Parameters Found: {search.best_params_}")

# 4. EVALUATION
preds_log = best_model.predict(X_test)
preds_egp = np.expm1(preds_log)
actuals_egp = np.expm1(y_test)

mae = mean_absolute_error(actuals_egp, preds_egp)
rmse = root_mean_squared_error(actuals_egp, preds_egp)
r2 = r2_score(y_test, preds_log)

print("\n" + "=" * 40)
print("  FINAL OPTIMIZED RESULTS")
print("=" * 40)
print(f"MAE:  {mae:,.0f} EGP")
print(f"RMSE: {rmse:,.0f} EGP")
print(f"R2:   {r2:.4f}")
print("=" * 40)

# 5. DIAGNOSTICS & DRIVERS
plt.figure(figsize=(10, 6))
sns.scatterplot(x=actuals_egp, y=preds_egp, alpha=0.5, color="teal")
plt.plot(
    [actuals_egp.min(), actuals_egp.max()],
    [actuals_egp.min(), actuals_egp.max()],
    "--r"
)
plt.title("Optimized Actual vs Predicted")
plt.savefig("optimized_performance.png")

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("Specific Salary Drivers (Optimized)")
plt.savefig("optimized_drivers.png")

# 6. SAVE
with open("salary_prediction_model/optimized_salary_model.pkl", "wb") as f:
    pickle.dump(best_model, f)